//! Substrate sandbox body-policy package.
//!
//! This crate models the body contract used by Substrate's capability sandbox
//! planning layer. It is a **policy and validation package only**: it does not
//! launch, supervise, download, inspect, or install anything.
//!
//! It is **not** a security sandbox by itself. A future broker may enforce
//! accepted plans; this package only defines and validates them.
//!
//! # Design constraints
//!
//! - Rust stable, standard library only
//! - No third-party crates, no build script, no `unsafe`, no FFI
//! - No process spawning, sockets, filesystem traversal, or credential handling
//! - No host-path mounts in the public typed API
//! - Dry-run validation always requires `execution_permitted == false`

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::BTreeSet;
use std::fmt;

/// Schema identifier for this body-policy contract.
///
/// Must remain exactly `substrate-sandbox-body-v1` for this crate revision.
pub const SCHEMA_VERSION: &str = "substrate-sandbox-body-v1";

/// Explicit capability grants a future adapter or broker may receive.
///
/// This is a closed set. Presence of a variant means the plan *names* a grant;
/// it does not authorize execution. Dry-run validation still requires
/// `execution_permitted == false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Capability {
    /// Outbound or general network access.
    Network,
    /// Writes outside the portable ephemeral/quarantine mounts.
    FilesystemWrite,
    /// Spawning or supervising child processes.
    Subprocess,
    /// Browser observation or automation surface.
    Browser,
    /// Media decode / frame or waveform consumption.
    MediaDecode,
    /// Nested containers or nested sandbox runtimes.
    ContainerNesting,
    /// GPU device access.
    Gpu,
    /// Microphone capture.
    Microphone,
    /// Camera capture.
    Camera,
    /// Desktop / GUI control surface.
    Desktop,
}

impl Capability {
    /// Stable snake-ish name for diagnostics and serialization by callers.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Network => "network",
            Self::FilesystemWrite => "filesystem_write",
            Self::Subprocess => "subprocess",
            Self::Browser => "browser",
            Self::MediaDecode => "media_decode",
            Self::ContainerNesting => "container_nesting",
            Self::Gpu => "gpu",
            Self::Microphone => "microphone",
            Self::Camera => "camera",
            Self::Desktop => "desktop",
        }
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Portable non-host mount sources.
///
/// Host paths cannot be expressed through this enum; that is intentional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum MountSource {
    /// Content-addressed task inputs (required access: read-only).
    ContentAddressedInputs,
    /// Scratch workspace for the task (required access: ephemeral-write).
    TaskWorkspace,
    /// Untrusted outputs isolated for review (required access: quarantine-write).
    UntrustedOutput,
}

impl MountSource {
    /// Canonical source identifier used by the body contract.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContentAddressedInputs => "content-addressed-inputs",
            Self::TaskWorkspace => "task-workspace",
            Self::UntrustedOutput => "untrusted-output",
        }
    }

    /// Access mode required for this portable source.
    pub const fn required_access(self) -> MountAccess {
        match self {
            Self::ContentAddressedInputs => MountAccess::ReadOnly,
            Self::TaskWorkspace => MountAccess::EphemeralWrite,
            Self::UntrustedOutput => MountAccess::QuarantineWrite,
        }
    }

    /// All portable sources, in contract order.
    pub const fn all() -> [MountSource; 3] {
        [
            Self::ContentAddressedInputs,
            Self::TaskWorkspace,
            Self::UntrustedOutput,
        ]
    }
}

impl fmt::Display for MountSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Access mode for a portable mount.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum MountAccess {
    /// Immutable inputs.
    ReadOnly,
    /// Task-local ephemeral write space.
    EphemeralWrite,
    /// Writes held in quarantine until review.
    QuarantineWrite,
}

impl MountAccess {
    /// Stable name for diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::EphemeralWrite => "ephemeral-write",
            Self::QuarantineWrite => "quarantine-write",
        }
    }
}

impl fmt::Display for MountAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A portable mount binding a source to an access mode.
///
/// Construct only through [`Mount::portable`] (correct access) or
/// [`Mount::with_access`] (for deliberate mismatch tests / planning). There is
/// no host-path constructor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Mount {
    source: MountSource,
    access: MountAccess,
}

impl Mount {
    /// Portable mount with the access mode required by `source`.
    pub const fn portable(source: MountSource) -> Self {
        Self {
            source,
            access: source.required_access(),
        }
    }

    /// Portable mount with an explicit access mode.
    ///
    /// Dry-run validation rejects pairings that do not match
    /// [`MountSource::required_access`]. Use this only when you need to
    /// represent or test an invalid plan; prefer [`Mount::portable`] otherwise.
    pub const fn with_access(source: MountSource, access: MountAccess) -> Self {
        Self { source, access }
    }

    /// Mount source.
    pub const fn source(&self) -> MountSource {
        self.source
    }

    /// Mount access mode.
    pub const fn access(&self) -> MountAccess {
        self.access
    }

    /// Whether access matches the source's required mode.
    pub const fn access_is_required(&self) -> bool {
        matches!(
            (self.source, self.access),
            (MountSource::ContentAddressedInputs, MountAccess::ReadOnly)
                | (MountSource::TaskWorkspace, MountAccess::EphemeralWrite)
                | (MountSource::UntrustedOutput, MountAccess::QuarantineWrite)
        )
    }
}

/// Strictly positive resource limits for a sandbox body plan.
///
/// Zero or negative-equivalent values are rejected by validation. Construction
/// does not enforce positivity so invalid plans can be tested; call
/// [`ResourceBudget::validate`] or plan validation before use.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceBudget {
    /// CPU allotment in milli-CPUs (or any positive integer unit the broker uses).
    pub cpu: u64,
    /// Memory limit in mebibytes.
    pub memory_mib: u64,
    /// Disk limit in mebibytes.
    pub disk_mib: u64,
    /// Maximum process count.
    pub processes: u64,
    /// Maximum thread count.
    pub threads: u64,
    /// Wall-clock time limit in seconds.
    pub wall_clock_seconds: u64,
    /// Maximum total output bytes.
    pub output_bytes: u64,
}

impl ResourceBudget {
    /// Build a budget. Does not check positivity; use [`Self::validate`].
    pub const fn new(
        cpu: u64,
        memory_mib: u64,
        disk_mib: u64,
        processes: u64,
        threads: u64,
        wall_clock_seconds: u64,
        output_bytes: u64,
    ) -> Self {
        Self {
            cpu,
            memory_mib,
            disk_mib,
            processes,
            threads,
            wall_clock_seconds,
            output_bytes,
        }
    }

    /// Conservative positive defaults suitable for dry-run planning.
    pub const fn dry_run_default() -> Self {
        Self {
            cpu: 1000,
            memory_mib: 512,
            disk_mib: 1024,
            processes: 32,
            threads: 128,
            wall_clock_seconds: 300,
            output_bytes: 64 * 1024 * 1024,
        }
    }

    /// Ensure every limit is strictly positive.
    pub fn validate(&self) -> Result<(), ValidationError> {
        let checks = [
            ("cpu", self.cpu),
            ("memory_mib", self.memory_mib),
            ("disk_mib", self.disk_mib),
            ("processes", self.processes),
            ("threads", self.threads),
            ("wall_clock_seconds", self.wall_clock_seconds),
            ("output_bytes", self.output_bytes),
        ];
        for (field, value) in checks {
            if value == 0 {
                return Err(ValidationError::InvalidResourceLimit { field });
            }
        }
        Ok(())
    }
}

/// Network posture for a body plan.
///
/// Dry-run validation accepts only [`NetworkMode::Deny`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum NetworkMode {
    /// No network / deny egress. Only mode accepted by dry-run validation.
    Deny,
    /// Planned restricted egress (not accepted under dry-run validation).
    Restricted,
    /// Planned open egress (not accepted under dry-run validation).
    AllowEgress,
}

impl NetworkMode {
    /// Stable name for diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Deny => "deny",
            Self::Restricted => "restricted",
            Self::AllowEgress => "allow_egress",
        }
    }

    /// Whether this mode is accepted by dry-run validation.
    pub const fn is_deny(self) -> bool {
        matches!(self, Self::Deny)
    }
}

impl fmt::Display for NetworkMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Closed tool **roles** (adapter kinds), not executable names or paths.
///
/// Each role declares the capability grants it requires. Operators later bind
/// a license-reviewed adapter binary to a role via signed packs; that binding
/// is out of scope here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ToolAdapterKind {
    /// Repository / tree inspection over portable inputs (no extra capabilities).
    RepositoryInspection,
    /// Browser observation surface; requires [`Capability::Browser`].
    BrowserObservation,
    /// Media observation (frames / waveforms); requires [`Capability::MediaDecode`].
    MediaObservation,
}

impl ToolAdapterKind {
    /// Stable role name (not a path or binary name).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RepositoryInspection => "repository_inspection",
            Self::BrowserObservation => "browser_observation",
            Self::MediaObservation => "media_observation",
        }
    }

    /// Capability grants this role requires when present on a plan.
    pub const fn required_capabilities(self) -> &'static [Capability] {
        match self {
            Self::RepositoryInspection => &[],
            Self::BrowserObservation => &[Capability::Browser],
            Self::MediaObservation => &[Capability::MediaDecode],
        }
    }
}

impl fmt::Display for ToolAdapterKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Typed dry-run / plan validation failure.
///
/// Validation is fail-closed: any of these variants means the plan must not be
/// treated as an accepted dry-run body plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValidationError {
    /// Plan schema string does not match [`SCHEMA_VERSION`].
    SchemaMismatch {
        /// Observed schema version on the plan.
        found: String,
    },
    /// Dry-run forbids execution authorization.
    ExecutionPermitted,
    /// Network mode is not deny / no-network.
    NetworkEgress {
        /// Observed network mode.
        mode: NetworkMode,
    },
    /// A required portable mount source is absent.
    MissingMount {
        /// Missing source.
        source: MountSource,
    },
    /// The same mount source appears more than once.
    DuplicateMount {
        /// Duplicated source.
        source: MountSource,
    },
    /// Mount access does not match the source's required mode.
    WrongMountAccess {
        /// Mount source.
        source: MountSource,
        /// Access that was declared.
        found: MountAccess,
        /// Access required by the contract.
        expected: MountAccess,
    },
    /// A resource limit is not strictly positive.
    InvalidResourceLimit {
        /// Field name that failed.
        field: &'static str,
    },
    /// A tool role is present but a required capability grant is missing.
    MissingToolCapabilityGrant {
        /// Tool role.
        tool: ToolAdapterKind,
        /// Capability that must be granted.
        capability: Capability,
    },
    /// A tool role requires a capability that is unapproved under dry-run policy.
    ///
    /// Dry-run only approves capabilities that appear in a tool role's
    /// `required_capabilities` list for roles that are themselves dry-run
    /// eligible. Roles whose requirements fall outside the dry-run allowlist
    /// fail here (none of the current closed roles do; this exists for
    /// fail-closed extension).
    UnapprovedToolCapability {
        /// Tool role.
        tool: ToolAdapterKind,
        /// Capability that is not approved under dry-run.
        capability: Capability,
    },
    /// Browser tool use without an explicit [`Capability::Browser`] grant.
    BrowserWithoutCapability,
    /// Media tool use without an explicit [`Capability::MediaDecode`] grant.
    MediaWithoutCapability,
    /// A grant is present that dry-run does not approve for this plan.
    ///
    /// Dry-run allows only capabilities required by declared tool roles (today:
    /// `Browser` and `MediaDecode` when those tools are present). Extra grants
    /// such as `Network`, `Subprocess`, or `Gpu` are refused.
    UnapprovedCapabilityGrant {
        /// Capability grant that is not approved.
        capability: Capability,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaMismatch { found } => {
                write!(
                    f,
                    "schema mismatch: found {found:?}, expected {SCHEMA_VERSION:?}"
                )
            }
            Self::ExecutionPermitted => {
                write!(f, "execution_permitted must be false for dry-run plans")
            }
            Self::NetworkEgress { mode } => {
                write!(
                    f,
                    "network egress refused under dry-run: mode is {mode}, only deny is accepted"
                )
            }
            Self::MissingMount { source } => {
                write!(f, "missing required portable mount source: {source}")
            }
            Self::DuplicateMount { source } => {
                write!(f, "duplicate mount source: {source}")
            }
            Self::WrongMountAccess {
                source,
                found,
                expected,
            } => {
                write!(
                    f,
                    "wrong mount access for {source}: found {found}, expected {expected}"
                )
            }
            Self::InvalidResourceLimit { field } => {
                write!(f, "resource limit {field} must be strictly positive")
            }
            Self::MissingToolCapabilityGrant { tool, capability } => {
                write!(
                    f,
                    "tool role {tool} requires capability grant {capability}, which is missing"
                )
            }
            Self::UnapprovedToolCapability { tool, capability } => {
                write!(
                    f,
                    "tool role {tool} implies unapproved capability {capability} under dry-run"
                )
            }
            Self::BrowserWithoutCapability => {
                write!(
                    f,
                    "browser tool use requires explicit capability grant: browser"
                )
            }
            Self::MediaWithoutCapability => {
                write!(
                    f,
                    "media tool use requires explicit capability grant: media_decode"
                )
            }
            Self::UnapprovedCapabilityGrant { capability } => {
                write!(
                    f,
                    "capability grant {capability} is not approved under dry-run for this plan"
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Capabilities that dry-run may approve when a declared tool role requires them.
const fn dry_run_eligible_capability(cap: Capability) -> bool {
    matches!(cap, Capability::Browser | Capability::MediaDecode)
}

/// Complete sandbox body plan (policy document).
///
/// Field visibility is intentional so callers can compose plans for validation.
/// Prefer [`SandboxBodyPlan::dry_run_default`] for a known-good starting point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxBodyPlan {
    /// Must equal [`SCHEMA_VERSION`] for acceptance.
    pub schema_version: String,
    /// Execution authorization. Dry-run validation requires `false`.
    pub execution_permitted: bool,
    /// Network posture. Dry-run validation requires [`NetworkMode::Deny`].
    pub network_mode: NetworkMode,
    /// Portable mounts. Must include each portable source exactly once with
    /// the required access mode.
    pub mounts: Vec<Mount>,
    /// Explicit capability grants.
    pub grants: Vec<Capability>,
    /// Strictly positive resource budget.
    pub resource_budget: ResourceBudget,
    /// Declared tool roles (adapter kinds), without paths or binary names.
    pub tool_roles: Vec<ToolAdapterKind>,
}

impl SandboxBodyPlan {
    /// Safe default dry-run plan:
    ///
    /// - schema [`SCHEMA_VERSION`]
    /// - all three portable mounts with required access
    /// - network deny
    /// - no capability grants
    /// - no tool roles
    /// - positive default resource budget
    /// - `execution_permitted == false`
    pub fn dry_run_default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            execution_permitted: false,
            network_mode: NetworkMode::Deny,
            mounts: MountSource::all()
                .into_iter()
                .map(Mount::portable)
                .collect(),
            grants: Vec::new(),
            resource_budget: ResourceBudget::dry_run_default(),
            tool_roles: Vec::new(),
        }
    }

    /// Fail-closed dry-run validation.
    ///
    /// On success the plan is a consistent policy document with execution
    /// still forbidden. On error the plan must not be treated as accepted.
    pub fn validate_dry_run(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ValidationError::SchemaMismatch {
                found: self.schema_version.clone(),
            });
        }

        if self.execution_permitted {
            return Err(ValidationError::ExecutionPermitted);
        }

        if !self.network_mode.is_deny() {
            return Err(ValidationError::NetworkEgress {
                mode: self.network_mode,
            });
        }

        self.validate_mounts()?;
        self.resource_budget.validate()?;
        self.validate_tools_and_grants()?;

        Ok(())
    }

    /// Alias for [`Self::validate_dry_run`].
    ///
    /// Kept so callers can use either name; both are fail-closed dry-run
    /// checks with no execution path.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.validate_dry_run()
    }

    fn validate_mounts(&self) -> Result<(), ValidationError> {
        let mut seen: BTreeSet<MountSource> = BTreeSet::new();

        for mount in &self.mounts {
            let source = mount.source();
            if !seen.insert(source) {
                return Err(ValidationError::DuplicateMount { source });
            }
            let expected = source.required_access();
            if mount.access() != expected {
                return Err(ValidationError::WrongMountAccess {
                    source,
                    found: mount.access(),
                    expected,
                });
            }
        }

        for source in MountSource::all() {
            if !seen.contains(&source) {
                return Err(ValidationError::MissingMount { source });
            }
        }

        Ok(())
    }

    fn validate_tools_and_grants(&self) -> Result<(), ValidationError> {
        let grant_set: BTreeSet<Capability> = self.grants.iter().copied().collect();

        // Tool roles: required capabilities must be granted and dry-run eligible.
        for tool in &self.tool_roles {
            for &capability in tool.required_capabilities() {
                if !dry_run_eligible_capability(capability) {
                    return Err(ValidationError::UnapprovedToolCapability {
                        tool: *tool,
                        capability,
                    });
                }
                if !grant_set.contains(&capability) {
                    // Specific browser/media errors first for clarity.
                    if matches!(*tool, ToolAdapterKind::BrowserObservation)
                        && capability == Capability::Browser
                    {
                        return Err(ValidationError::BrowserWithoutCapability);
                    }
                    if matches!(*tool, ToolAdapterKind::MediaObservation)
                        && capability == Capability::MediaDecode
                    {
                        return Err(ValidationError::MediaWithoutCapability);
                    }
                    return Err(ValidationError::MissingToolCapabilityGrant {
                        tool: *tool,
                        capability,
                    });
                }
            }
        }

        // Grants must be justified by declared tool roles under dry-run.
        let mut allowed: BTreeSet<Capability> = BTreeSet::new();
        for tool in &self.tool_roles {
            for &capability in tool.required_capabilities() {
                if dry_run_eligible_capability(capability) {
                    allowed.insert(capability);
                }
            }
        }

        for &capability in &grant_set {
            if !allowed.contains(&capability) {
                return Err(ValidationError::UnapprovedCapabilityGrant { capability });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_version_is_exact() {
        assert_eq!(SCHEMA_VERSION, "substrate-sandbox-body-v1");
    }

    #[test]
    fn default_dry_run_plan_is_valid() {
        let plan = SandboxBodyPlan::dry_run_default();
        assert_eq!(plan.schema_version, SCHEMA_VERSION);
        assert!(!plan.execution_permitted);
        assert_eq!(plan.network_mode, NetworkMode::Deny);
        assert!(plan.grants.is_empty());
        assert!(plan.tool_roles.is_empty());
        assert_eq!(plan.mounts.len(), 3);
        plan.validate_dry_run().expect("default must validate");
        plan.validate().expect("validate aliases dry-run");
    }

    #[test]
    fn default_mounts_are_portable_with_required_access() {
        let plan = SandboxBodyPlan::dry_run_default();
        let expected = [
            (MountSource::ContentAddressedInputs, MountAccess::ReadOnly),
            (MountSource::TaskWorkspace, MountAccess::EphemeralWrite),
            (MountSource::UntrustedOutput, MountAccess::QuarantineWrite),
        ];
        for (mount, (source, access)) in plan.mounts.iter().zip(expected) {
            assert_eq!(mount.source(), source);
            assert_eq!(mount.access(), access);
            assert!(mount.access_is_required());
        }
    }

    #[test]
    fn host_path_mount_cannot_be_expressed() {
        // Compile-time / API surface check: MountSource is closed over portable
        // identifiers only. Runtime string check for documentation stability.
        let names: Vec<&str> = MountSource::all().iter().map(|s| s.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "content-addressed-inputs",
                "task-workspace",
                "untrusted-output",
            ]
        );
        assert!(!names.iter().any(|n| n.contains('/') || n.contains('\\')));
    }

    #[test]
    fn rejects_execution_permitted() {
        let mut plan = SandboxBodyPlan::dry_run_default();
        plan.execution_permitted = true;
        assert_eq!(
            plan.validate_dry_run(),
            Err(ValidationError::ExecutionPermitted)
        );
    }

    #[test]
    fn rejects_network_egress_modes() {
        for mode in [NetworkMode::Restricted, NetworkMode::AllowEgress] {
            let mut plan = SandboxBodyPlan::dry_run_default();
            plan.network_mode = mode;
            assert_eq!(
                plan.validate_dry_run(),
                Err(ValidationError::NetworkEgress { mode })
            );
        }
    }

    #[test]
    fn rejects_schema_mismatch() {
        let mut plan = SandboxBodyPlan::dry_run_default();
        plan.schema_version = "not-the-schema".into();
        assert_eq!(
            plan.validate_dry_run(),
            Err(ValidationError::SchemaMismatch {
                found: "not-the-schema".into()
            })
        );
    }

    #[test]
    fn rejects_missing_mount() {
        let mut plan = SandboxBodyPlan::dry_run_default();
        plan.mounts
            .retain(|m| m.source() != MountSource::UntrustedOutput);
        assert_eq!(
            plan.validate_dry_run(),
            Err(ValidationError::MissingMount {
                source: MountSource::UntrustedOutput
            })
        );
    }

    #[test]
    fn rejects_duplicate_mount() {
        let mut plan = SandboxBodyPlan::dry_run_default();
        plan.mounts
            .push(Mount::portable(MountSource::TaskWorkspace));
        assert_eq!(
            plan.validate_dry_run(),
            Err(ValidationError::DuplicateMount {
                source: MountSource::TaskWorkspace
            })
        );
    }

    #[test]
    fn rejects_wrong_mount_access() {
        let mut plan = SandboxBodyPlan::dry_run_default();
        plan.mounts = vec![
            Mount::with_access(
                MountSource::ContentAddressedInputs,
                MountAccess::EphemeralWrite,
            ),
            Mount::portable(MountSource::TaskWorkspace),
            Mount::portable(MountSource::UntrustedOutput),
        ];
        assert_eq!(
            plan.validate_dry_run(),
            Err(ValidationError::WrongMountAccess {
                source: MountSource::ContentAddressedInputs,
                found: MountAccess::EphemeralWrite,
                expected: MountAccess::ReadOnly,
            })
        );
    }

    #[test]
    fn rejects_zero_resource_limits() {
        let fields = [
            "cpu",
            "memory_mib",
            "disk_mib",
            "processes",
            "threads",
            "wall_clock_seconds",
            "output_bytes",
        ];
        for field in fields {
            let mut plan = SandboxBodyPlan::dry_run_default();
            match field {
                "cpu" => plan.resource_budget.cpu = 0,
                "memory_mib" => plan.resource_budget.memory_mib = 0,
                "disk_mib" => plan.resource_budget.disk_mib = 0,
                "processes" => plan.resource_budget.processes = 0,
                "threads" => plan.resource_budget.threads = 0,
                "wall_clock_seconds" => plan.resource_budget.wall_clock_seconds = 0,
                "output_bytes" => plan.resource_budget.output_bytes = 0,
                _ => unreachable!(),
            }
            assert_eq!(
                plan.validate_dry_run(),
                Err(ValidationError::InvalidResourceLimit { field })
            );
        }
    }

    #[test]
    fn rejects_browser_without_capability() {
        let mut plan = SandboxBodyPlan::dry_run_default();
        plan.tool_roles = vec![ToolAdapterKind::BrowserObservation];
        assert_eq!(
            plan.validate_dry_run(),
            Err(ValidationError::BrowserWithoutCapability)
        );
    }

    #[test]
    fn rejects_media_without_capability() {
        let mut plan = SandboxBodyPlan::dry_run_default();
        plan.tool_roles = vec![ToolAdapterKind::MediaObservation];
        assert_eq!(
            plan.validate_dry_run(),
            Err(ValidationError::MediaWithoutCapability)
        );
    }

    #[test]
    fn accepts_browser_and_media_with_explicit_grants() {
        let mut plan = SandboxBodyPlan::dry_run_default();
        plan.tool_roles = vec![
            ToolAdapterKind::RepositoryInspection,
            ToolAdapterKind::BrowserObservation,
            ToolAdapterKind::MediaObservation,
        ];
        plan.grants = vec![Capability::Browser, Capability::MediaDecode];
        plan.validate_dry_run().expect("tools with grants");
    }

    #[test]
    fn rejects_unapproved_capability_grants() {
        let mut plan = SandboxBodyPlan::dry_run_default();
        plan.grants = vec![Capability::Network];
        assert_eq!(
            plan.validate_dry_run(),
            Err(ValidationError::UnapprovedCapabilityGrant {
                capability: Capability::Network
            })
        );

        let mut plan = SandboxBodyPlan::dry_run_default();
        plan.tool_roles = vec![ToolAdapterKind::BrowserObservation];
        plan.grants = vec![Capability::Browser, Capability::Subprocess];
        assert_eq!(
            plan.validate_dry_run(),
            Err(ValidationError::UnapprovedCapabilityGrant {
                capability: Capability::Subprocess
            })
        );
    }

    #[test]
    fn tool_roles_state_capabilities_not_paths() {
        for tool in [
            ToolAdapterKind::RepositoryInspection,
            ToolAdapterKind::BrowserObservation,
            ToolAdapterKind::MediaObservation,
        ] {
            let name = tool.as_str();
            assert!(!name.contains('/'));
            assert!(!name.contains('\\'));
            assert!(!name.ends_with(".exe"));
            // Roles declare required capabilities; they never name a binary.
            let _ = tool.required_capabilities();
        }
        assert_eq!(
            ToolAdapterKind::BrowserObservation.required_capabilities(),
            &[Capability::Browser]
        );
        assert_eq!(
            ToolAdapterKind::MediaObservation.required_capabilities(),
            &[Capability::MediaDecode]
        );
        assert!(ToolAdapterKind::RepositoryInspection
            .required_capabilities()
            .is_empty());
    }

    #[test]
    fn capability_set_is_closed_and_complete() {
        let all = [
            Capability::Network,
            Capability::FilesystemWrite,
            Capability::Subprocess,
            Capability::Browser,
            Capability::MediaDecode,
            Capability::ContainerNesting,
            Capability::Gpu,
            Capability::Microphone,
            Capability::Camera,
            Capability::Desktop,
        ];
        assert_eq!(all.len(), 10);
        for cap in all {
            assert!(!cap.as_str().is_empty());
        }
    }

    #[test]
    fn resource_budget_validate_direct() {
        assert!(ResourceBudget::dry_run_default().validate().is_ok());
        assert_eq!(
            ResourceBudget::new(0, 1, 1, 1, 1, 1, 1).validate(),
            Err(ValidationError::InvalidResourceLimit { field: "cpu" })
        );
    }

    #[test]
    fn error_display_is_informative() {
        let err = ValidationError::ExecutionPermitted;
        assert!(err.to_string().contains("execution_permitted"));
    }
}

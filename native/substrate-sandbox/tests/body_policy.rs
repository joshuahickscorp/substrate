//! Integration tests for the sandbox body-policy package.
//!
//! These exercise the public API as an external crate would: default plan
//! acceptance and every stated dry-run refusal.

use substrate_sandbox::{
    Capability, Mount, MountAccess, MountSource, NetworkMode, ResourceBudget, SandboxBodyPlan,
    ToolAdapterKind, ValidationError, SCHEMA_VERSION,
};

fn base() -> SandboxBodyPlan {
    SandboxBodyPlan::dry_run_default()
}

#[test]
fn default_plan_is_accepted_and_forbids_execution() {
    let plan = base();
    assert_eq!(plan.schema_version, SCHEMA_VERSION);
    assert_eq!(SCHEMA_VERSION, "substrate-sandbox-body-v1");
    assert!(!plan.execution_permitted);
    assert_eq!(plan.network_mode, NetworkMode::Deny);
    assert!(plan.grants.is_empty());
    assert!(plan.tool_roles.is_empty());
    plan.validate_dry_run().expect("default dry-run plan");
    plan.validate().expect("validate is an alias");
}

#[test]
fn refusal_execution_permitted() {
    let mut plan = base();
    plan.execution_permitted = true;
    assert_eq!(
        plan.validate_dry_run(),
        Err(ValidationError::ExecutionPermitted)
    );
}

#[test]
fn refusal_network_egress() {
    for mode in [NetworkMode::Restricted, NetworkMode::AllowEgress] {
        let mut plan = base();
        plan.network_mode = mode;
        assert_eq!(
            plan.validate_dry_run(),
            Err(ValidationError::NetworkEgress { mode })
        );
    }
}

#[test]
fn refusal_schema_mismatch() {
    let mut plan = base();
    plan.schema_version = "substrate-sandbox-body-v0".into();
    assert!(matches!(
        plan.validate_dry_run(),
        Err(ValidationError::SchemaMismatch { .. })
    ));
}

#[test]
fn refusal_missing_mount_source() {
    let mut plan = base();
    plan.mounts.clear();
    plan.mounts
        .push(Mount::portable(MountSource::TaskWorkspace));
    plan.mounts
        .push(Mount::portable(MountSource::UntrustedOutput));
    assert_eq!(
        plan.validate_dry_run(),
        Err(ValidationError::MissingMount {
            source: MountSource::ContentAddressedInputs
        })
    );
}

#[test]
fn refusal_duplicate_mount_source() {
    let mut plan = base();
    plan.mounts
        .push(Mount::portable(MountSource::ContentAddressedInputs));
    assert_eq!(
        plan.validate_dry_run(),
        Err(ValidationError::DuplicateMount {
            source: MountSource::ContentAddressedInputs
        })
    );
}

#[test]
fn refusal_wrong_mount_access() {
    let mut plan = base();
    plan.mounts = vec![
        Mount::portable(MountSource::ContentAddressedInputs),
        Mount::with_access(MountSource::TaskWorkspace, MountAccess::ReadOnly),
        Mount::portable(MountSource::UntrustedOutput),
    ];
    assert_eq!(
        plan.validate_dry_run(),
        Err(ValidationError::WrongMountAccess {
            source: MountSource::TaskWorkspace,
            found: MountAccess::ReadOnly,
            expected: MountAccess::EphemeralWrite,
        })
    );
}

#[test]
fn refusal_invalid_resource_limits() {
    let mut plan = base();
    plan.resource_budget = ResourceBudget::new(1, 1, 1, 1, 1, 1, 0);
    assert_eq!(
        plan.validate_dry_run(),
        Err(ValidationError::InvalidResourceLimit {
            field: "output_bytes"
        })
    );
}

#[test]
fn refusal_browser_without_capability() {
    let mut plan = base();
    plan.tool_roles = vec![ToolAdapterKind::BrowserObservation];
    assert_eq!(
        plan.validate_dry_run(),
        Err(ValidationError::BrowserWithoutCapability)
    );
}

#[test]
fn refusal_media_without_capability() {
    let mut plan = base();
    plan.tool_roles = vec![ToolAdapterKind::MediaObservation];
    assert_eq!(
        plan.validate_dry_run(),
        Err(ValidationError::MediaWithoutCapability)
    );
}

#[test]
fn refusal_missing_tool_capability_grant_is_covered_by_browser_media() {
    // Browser/media roles are the current tools that require grants. Missing
    // those grants is refused (specialized errors). Repository inspection needs
    // none and remains valid without grants.
    let mut plan = base();
    plan.tool_roles = vec![ToolAdapterKind::RepositoryInspection];
    plan.validate_dry_run()
        .expect("repository inspection requires no grants");

    plan.tool_roles = vec![ToolAdapterKind::BrowserObservation];
    assert!(plan.validate_dry_run().is_err());
}

#[test]
fn refusal_unapproved_capability_grant() {
    // Extra grants not justified by tool roles are unapproved under dry-run.
    let mut plan = base();
    plan.grants = vec![Capability::Gpu];
    assert_eq!(
        plan.validate_dry_run(),
        Err(ValidationError::UnapprovedCapabilityGrant {
            capability: Capability::Gpu
        })
    );
}

#[test]
fn refusal_tool_implies_unapproved_when_grant_not_dry_run_eligible() {
    // Declared tool roles may only pull dry-run-eligible capabilities
    // (Browser, MediaDecode). Unapproved grants paired with tools still fail.
    let mut plan = base();
    plan.tool_roles = vec![ToolAdapterKind::BrowserObservation];
    plan.grants = vec![Capability::Browser, Capability::Network];
    assert_eq!(
        plan.validate_dry_run(),
        Err(ValidationError::UnapprovedCapabilityGrant {
            capability: Capability::Network
        })
    );
}

#[test]
fn accepted_plan_with_tools_and_matching_grants() {
    let mut plan = base();
    plan.tool_roles = vec![
        ToolAdapterKind::RepositoryInspection,
        ToolAdapterKind::BrowserObservation,
        ToolAdapterKind::MediaObservation,
    ];
    plan.grants = vec![Capability::Browser, Capability::MediaDecode];
    plan.validate_dry_run().expect("matching grants");
    assert!(!plan.execution_permitted);
}

#[test]
fn tool_roles_are_roles_not_binaries() {
    assert_eq!(
        ToolAdapterKind::RepositoryInspection.as_str(),
        "repository_inspection"
    );
    assert_eq!(
        ToolAdapterKind::BrowserObservation.as_str(),
        "browser_observation"
    );
    assert_eq!(
        ToolAdapterKind::MediaObservation.as_str(),
        "media_observation"
    );
    assert_eq!(
        ToolAdapterKind::BrowserObservation.required_capabilities(),
        &[Capability::Browser]
    );
    assert_eq!(
        ToolAdapterKind::MediaObservation.required_capabilities(),
        &[Capability::MediaDecode]
    );
}

#[test]
fn mount_api_exposes_only_portable_sources() {
    for source in MountSource::all() {
        let mount = Mount::portable(source);
        assert!(mount.access_is_required());
        assert_eq!(mount.access(), source.required_access());
    }
}

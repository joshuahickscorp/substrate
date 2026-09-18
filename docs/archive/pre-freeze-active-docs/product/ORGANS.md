# Organs

An organ is a replaceable reasoning or specialist-computation interface. It
may eventually be a local or remote language model, vision model, speech
model, formal solver, simulator, or typed tool adapter.

The current entity contract records only an organ identifier, interface
version, and supported modalities. It deliberately stores no model weights,
provider credentials, hidden conversation history, or unrestricted tool
authority. Replacing an organ must preserve entity identity, receipts,
projects, and verified developmental state.

The organ gateway itself is not implemented yet. Future work must make each
organ invocation typed, budgeted, provenance-bearing, and independently
auditable before its output can be assimilated.


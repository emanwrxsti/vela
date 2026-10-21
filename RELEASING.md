# Release licensing checks

Before publishing source or binaries:

- Read `LICENSE` and `NOTICE` from the exact source commit being released.
  Current project code uses the Suprnova Vela Source-Available License 1.0.
  Do not substitute MIT or another license that permits commercial or
  closed-source derivative miners without an explicit owner instruction.
- Check that the release notes identify the applicable project license and
  the requirement for written permission for commercial or closed-source
  incorporation. Public source availability does not grant that permission.
- Confirm every Linux, HiveOS, MMPOS, SMOS, Windows, and Docker package contains
  that commit's `LICENSE` and `NOTICE`. Existing packaging scripts copy both;
  inspect the resulting archives and image as part of release verification.
- Preserve third-party license notices, including nlohmann/json and NVIDIA
  runtime notices. The project's license does not replace those licenses.
- Keep historical tags and released artifacts under their original terms.
  Releases through v0.1.4 were distributed under MIT; do not claim that a later
  license change revokes their permissions or silently replace their files.

The 2026-10-21 license change is a source/documentation update. It does not
change the mining code, bump the miner version, or replace the v0.1.4 binaries.

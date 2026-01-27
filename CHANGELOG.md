# [3.0.0](https://github.com/terrylica/trading-fitness/compare/v2.6.0...v3.0.0) (2026-01-27)


### Bug Fixes

* **deps:** add pandera to dev group for test collection ([866c943](https://github.com/terrylica/trading-fitness/commit/866c943a4a0362e6881ad9ddf0e44c1f71e7a22d))


### Features

* **deps:** upgrade rangebar-py to v11.0.0 with Ouroboros support ([7eba53b](https://github.com/terrylica/trading-fitness/commit/7eba53b560bf77c5f312a6cf0d24d54697a82498))
* **mise:** add preflight:test-collection to catch missing dev deps ([f6dff72](https://github.com/terrylica/trading-fitness/commit/f6dff7278a800773d931c5c637fbdae98e970246))


### BREAKING CHANGES

* **deps:** get_range_bars() now has ouroboros parameter (default: "year").
Existing code works unchanged with yearly reset boundaries.

Note: Existing cached range bars on bigblack will need reconstruction once
Ouroboros is enabled for production use.

ADR: docs/adr/2026-01-27-symmetric-dogfooding-rangebar.md

SRED-Type: support-work
SRED-Claim: TRADING-FITNESS

# [2.6.0](https://github.com/terrylica/trading-fitness/compare/v2.5.0...v2.6.0) (2026-01-27)


### Features

* **ith-python:** add walk-forward optimization module ([3a5465f](https://github.com/terrylica/trading-fitness/commit/3a5465ff334ab1fd13631a82e9773703922bed7a))
* **ith-python:** expand mise tasks for forensic pipeline ([db25dd3](https://github.com/terrylica/trading-fitness/commit/db25dd3c65b07ec9b10f8ebdc98d952726929f98))
* **scripts:** add range bar precompute scripts for bigblack ([bf2aaa0](https://github.com/terrylica/trading-fitness/commit/bf2aaa024737d923ba69debf0842a0640bad0aca))

# [2.5.0](https://github.com/terrylica/trading-fitness/compare/v2.4.0...v2.5.0) (2026-01-27)


### Features

* **data-layer:** add DAG-based data preparation pipeline ([20a4070](https://github.com/terrylica/trading-fitness/commit/20a4070b94f043ab78a89b43b5955ffe5bc214c5))
* **ith-python:** add ClaSPy feature extraction for regime detection ([7f00a82](https://github.com/terrylica/trading-fitness/commit/7f00a82732662b41c4b4e426064be84bd8a2475b))
* **ith-python:** add telemetry module for scientific reproducibility ([7cea57f](https://github.com/terrylica/trading-fitness/commit/7cea57f822662e4a6622166b16ad6e02a9ea026f))
* **ith-python:** implement multi-view feature architecture ([51c8eda](https://github.com/terrylica/trading-fitness/commit/51c8edaa0863de00185836acf137672ed38c6ebc))
* **validation:** add symmetric dogfooding with rangebar-py ([e4d1879](https://github.com/terrylica/trading-fitness/commit/e4d1879f28e86ebe775bd9dac00e245656333530))

# [2.4.0](https://github.com/terrylica/trading-fitness/compare/v2.3.0...v2.4.0) (2026-01-25)


### Features

* **ith-python:** add ClickHouse fixture for on-demand cache testing ([0dc100c](https://github.com/terrylica/trading-fitness/commit/0dc100c488666eae20f915bc6a284bc5c3ab57c3))
* **ith-python:** add statistical examination framework with rectified methods ([91a38ee](https://github.com/terrylica/trading-fitness/commit/91a38eeaff00c969e6486cabe75b950d2cab5831))

# [2.3.0](https://github.com/terrylica/trading-fitness/compare/v2.2.0...v2.3.0) (2026-01-22)


### Bug Fixes

* **ith-python:** fix test edge cases for symmetric dogfooding validation ([7812949](https://github.com/terrylica/trading-fitness/commit/7812949c86746d3a8f894a9015b2d2ec02074692))


### Features

* **metrics-rust:** add rolling ITH features with symmetric dogfooding validation ([6467721](https://github.com/terrylica/trading-fitness/commit/64677211354743d42b44cbc224fcbe42c099485d))

# [2.2.0](https://github.com/terrylica/trading-fitness/compare/v2.1.0...v2.2.0) (2026-01-21)


### Bug Fixes

* **ith-python:** constrain Python version to 3.12 ([be2eeaf](https://github.com/terrylica/trading-fitness/commit/be2eeafe586b757fd2a86389fd58d09df87b62bc))


### Features

* **ith-python:** add cross-validation utilities ([497f6ac](https://github.com/terrylica/trading-fitness/commit/497f6ac4c568895eda67362783617438cb44f58f))
* **ith-python:** add package-level mise.toml ([9a766ff](https://github.com/terrylica/trading-fitness/commit/9a766ffb4f1328949afe6d0749ae1dfc1f9ba8f5))
* **metrics-rust:** add proptest property-based tests ([86d6acd](https://github.com/terrylica/trading-fitness/commit/86d6acdcffbf14081d2a134be0aeae2e78eb8675))
* **metrics-rust:** improve ITH implementation and add proptest dep ([9ae5fb6](https://github.com/terrylica/trading-fitness/commit/9ae5fb664a053a50b95fa3fe55b5a769cb98d7ba))
* **mise:** add metrics-rust tasks and orchestration ([3556520](https://github.com/terrylica/trading-fitness/commit/35565201dfb9225be0d60b1a08987b88ab67fd6a))


### Performance Improvements

* **ith-python:** add batch NAV generation with Numba parallel ([b67e6a4](https://github.com/terrylica/trading-fitness/commit/b67e6a450856b778e579cb8c9f5fa5856d382b43))

# [2.1.0](https://github.com/terrylica/trading-fitness/compare/v2.0.0...v2.1.0) (2026-01-20)


### Features

* **metrics-rust:** add PyO3 Python bindings for BiLSTM metrics ([705d02c](https://github.com/terrylica/trading-fitness/commit/705d02c08c62623278c9b65eb9b297c543827a71))

# [2.0.0](https://github.com/terrylica/trading-fitness/compare/v1.1.0...v2.0.0) (2026-01-19)


### Features

* **metrics-rust:** time-agnostic BiLSTM metrics library ([1bff411](https://github.com/terrylica/trading-fitness/commit/1bff41153fe1abdb6490d2dda192238b0faf09f5))


### BREAKING CHANGES

* **metrics-rust:** New Rust workspace member requires Cargo.toml at root

SRED-Type: experimental-development
SRED-Claim: ITH-RANGEBAR-METRICS

# [1.1.0](https://github.com/terrylica/trading-fitness/compare/v1.0.5...v1.1.0) (2026-01-19)


### Features

* **ith:** time-agnostic ITH refactoring for universal applicability ([33d918a](https://github.com/terrylica/trading-fitness/commit/33d918a6cf1b248216b513a6a31b66bdfcc7ad18))

## [1.0.5](https://github.com/terrylica/trading-fitness/compare/v1.0.4...v1.0.5) (2026-01-19)


### Bug Fixes

* **mise:** run analyze:bull and analyze:bear sequentially ([306c36c](https://github.com/terrylica/trading-fitness/commit/306c36c7543a42b4b45520d08efae3f0ebe7e978))

## [1.0.4](https://github.com/terrylica/trading-fitness/compare/v1.0.3...v1.0.4) (2026-01-19)


### Bug Fixes

* **mise:** add missing bear analysis to analyze task ([2d4cc5c](https://github.com/terrylica/trading-fitness/commit/2d4cc5c2664cfcfdc5c3f912321c55b6fb2e6d03))

## [1.0.3](https://github.com/terrylica/trading-fitness/compare/v1.0.2...v1.0.3) (2026-01-19)

## [1.0.2](https://github.com/terrylica/trading-fitness/compare/v1.0.1...v1.0.2) (2026-01-19)

## [1.0.1](https://github.com/terrylica/trading-fitness/compare/v1.0.0...v1.0.1) (2026-01-19)


### Bug Fixes

* **ith:** enforce bull/bear algorithm symmetry ([d46d64f](https://github.com/terrylica/trading-fitness/commit/d46d64f43a6ca31651504f5233e9b5682d45bc7a))
* **release:** remove duplicate preflight dependency from version task ([a9c600b](https://github.com/terrylica/trading-fitness/commit/a9c600b018b956ce097edbe38e6b1e63ab508045))

# 1.0.0 (2026-01-19)


### Bug Fixes

* **auth:** prevent GitHub API rate limiting issues ([9d54a95](https://github.com/terrylica/trading-fitness/commit/9d54a95076244fa6f18b85f383f59021c3f5cebd))
* complete migration with missing files and build config ([eff1be7](https://github.com/terrylica/trading-fitness/commit/eff1be7df50d8deefeb6bed1b4186b1256f42936))


### Features

* complete polyglot implementation with tests and tooling ([8d6c9f2](https://github.com/terrylica/trading-fitness/commit/8d6c9f2b53ccae72f1f32b6a06ea64f54826ea9f))
* initialize trading-fitness polyglot monorepo ([811f427](https://github.com/terrylica/trading-fitness/commit/811f4274f33e4dfecebe2cb3d172e0b3ed544da0))
* **ith:** add Bear ITH analysis for short position profitability ([e7dcdce](https://github.com/terrylica/trading-fitness/commit/e7dcdce2fe9289aac317f7ee23ee63db7a1b90a8))

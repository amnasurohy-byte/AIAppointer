# Changelog

All notable changes to AI Appointer Assist will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-08

### Added
- Initial production release
- Employee Lookup mode for individual predictions
- Career Simulation mode for hypothetical scenarios
- Billet Lookup mode for reverse candidate search
- Directional rank flexibility (separate promotion/demotion sliders)
- Probability boost for promotion roles
- Rich explanations for predictions (rank progression, branch fit, training match)
- Constraint-based filtering (rank, branch, pool)
- Repetition penalty for role recommendations
- Smart role selection in simulation (filtered by rank/branch)
- Auto-calculated years of service based on rank
- Name and Branch columns in Billet Lookup results
- Comprehensive documentation (README, DEPLOYMENT, ARCHITECTURE)
- Production-ready file structure
- Air-gapped deployment support
- GitHub deployment configuration
- Environment variable configuration
- Deployment scripts (install.sh, start.sh, stop.sh)
- Model training script
- .gitignore for sensitive data
- requirements.txt for dependencies

### Changed
- Application name from "Starfleet Officer Appointment Predictor" to "AI Appointer Assist"
- Dataset path to use data/ directory
- Reorganized files into production structure (src/, scripts/, tests/, dev/, docs/)
- Improved UI responsiveness and text wrapping
- Enhanced constraint enforcement logic

### Fixed
- Rank flexibility now works bidirectionally (up and down)
- Promotion roles now appear in predictions with boost
- Confidence display shows correct percentages
- UI text cutoff issues resolved
- Cache clearing functionality added

### Security
- No hardcoded secrets
- Environment variable configuration
- Local-only data processing
- Input validation
- Error handling

## [0.3.0] - 2025-12-07

### Added
- Billet Lookup feature
- Rich explanations
- Training history integration
- Rank flexibility slider

### Changed
- Improved simulation mode UX
- Enhanced constraint generation

### Fixed
- Constraint enforcement issues
- UI responsiveness

## [0.2.0] - 2025-12-06

### Added
- Simulation mode
- Constraint-based filtering
- Repetition penalty

### Changed
- Model training pipeline
- Feature engineering

## [0.1.0] - 2025-12-05

### Added
- Initial prototype
- LightGBM model
- Basic UI
- Employee lookup

---

[1.0.0]: https://github.com/yourorg/ai-appointer/releases/tag/v1.0.0
[0.3.0]: https://github.com/yourorg/ai-appointer/releases/tag/v0.3.0
[0.2.0]: https://github.com/yourorg/ai-appointer/releases/tag/v0.2.0
[0.1.0]: https://github.com/yourorg/ai-appointer/releases/tag/v0.1.0

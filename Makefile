.PHONY: test-matrix test-baseline test-scaling test-memory analyze help

# Run full configuration matrix
test-matrix:
	python scripts/run_matrix_tests.py

# Run only baseline configuration
test-baseline:
	pytest tests/test_config_matrix.py::TestConfigurationMatrix::test_configuration[baseline] -v

# Run scaling tests (batch size and concurrency)
test-scaling:
	pytest tests/test_config_matrix.py::TestConfigurationMatrix::test_configuration[large_batch] -v
	pytest tests/test_config_matrix.py::TestConfigurationMatrix::test_configuration[high_concurrency] -v

# Run memory limit tests (larger models)
test-memory:
	pytest tests/test_config_matrix.py::TestConfigurationMatrix::test_configuration[7b_model] -v
	pytest tests/test_config_matrix.py::TestConfigurationMatrix::test_configuration[13b_model] -v

# Analyze existing results
analyze:
	pytest tests/test_config_matrix.py::TestMatrixAnalysis::test_analyze_matrix_results -v -s

# Clean test artifacts
clean:
	rm -rf .pytest_cache
	rm -rf matrix_results/
	rm -f matrix_report.html
	find . -type d -name "__pycache__" -exec rm -rf {} +

help:
	@echo "Configuration Matrix Testing Commands:"
	@echo "  make test-matrix    - Run full configuration matrix"
	@echo "  make test-baseline  - Run only baseline configuration"
	@echo "  make test-scaling   - Run batch size and concurrency tests"
	@echo "  make test-memory    - Run larger model tests"
	@echo "  make analyze        - Analyze existing results"
	@echo "  make clean          - Clean test artifacts"
	@echo ""
	@echo "View HTML report:"
	@echo "  open matrix_report.html"
	@echo ""
	@echo "Example usage:"
	@echo "  make test-baseline && make analyze"

# View HTML report
open matrix_report.html
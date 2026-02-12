PY_DIRS=CA2_Real_Time_Streaming_Kafka/codes CA3_Advanced_ML_Regression_RecSys/codes Data_Science_Final_Project/phase3/src Data_Science_Final_Project/phase3/scripts Data_Science_Final_Project/phase3/pipeline.py

.PHONY: fmt lint fmt-check lint-fix

fmt:
	black $(PY_DIRS)

fmt-check:
	black --check $(PY_DIRS)

lint:
	ruff check $(PY_DIRS)

lint-fix:
	ruff check --fix $(PY_DIRS)

# Makefile for a Python project

# 設定 Python 命令
PYTHON=python3

help:
	@echo "+------------------------- command manual ----------------------------------+"
	@echo "|         command         |                   description                   |"
	@echo "|-------------------------+-------------------------------------------------|"
	@echo "| help                    | show command manual                             |"
	@echo "| geometric-transform     | do geometric transform                          |"
	@echo "| find-contours           | isolate every workpieces in image               |"
	@echo "| bayesian_convolutional_transformer   | generate glcm feature from the workpiece images |"
	@echo "| bayesian_convolutional_transformer_special     | view computed tomography via dicom viewer       |"
	@echo "| test             | train model to predict material property        |"
	@echo "+---------------------------------------------------------------------------+"

# # 安裝依賴
# install:
# 	pip install -r requirements.txt

# 執行應用
bayesian_convolutional_transformer:
	$(PYTHON) BayConvT.py

bayesian_convolutional_transformer_special:
	$(PYTHON) BayConvT_N.py

test:
	$(PYTHON) BayConvT_test.py

# 清理編譯生成的文件
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete


# 偽目標
.PHONY: bayesian_convolutional_transformer bayesian_convolutional_transformer_special test clean run
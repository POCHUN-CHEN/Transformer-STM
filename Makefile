# 設定 Python 命令
PYTHON=python3

# ANSI 顏色碼
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
BLUE=\033[0;34m
NC=\033[0m # 無顏色

# # 自動抓取終端機名稱
# TERMINAL_NAME=$(shell xdotool getwindowfocus getwindowname)

# # 自動切換全螢幕
# fullscreen:
# 	wmctrl -r "$(TERMINAL_NAME)" -b toggle,fullscreen

help:
	@echo "+------------------------- command manual ----------------------------------------------------------+"
	@echo "|         command                 |                           description                           |"
	@echo "|---------------------------------+-----------------------------------------------------------------|"
	@echo "| help                            | show command manual                                             |"
	@echo "|                                                                                                   |"
	@echo "|[Prepare]                                                                                          |"
	@echo "| Pick_up_datas                   |Pick up original data out of STD                                 |"
	@echo "|                                                                                                   |"
	@echo "|[Train]                                                                                            |"
	@echo "| Train_CvT_model                 | $(GREEN)(Recommand)$(NC) Training Cvt modle with images & parameters inputs |"
	@echo "| Train_CvT_model_images          | Training Cvt modle with only images input                       |"
	@echo "| Train_FFN_model                 | Training FFN modle with only parameters input                   |"
	@echo "|                                                                                                   |"
	@echo "|[Test]                                                                                             |"
	@echo "| Test_CvT_model                  | Test Cvt modle with images & parameters inputs                  |"
	@echo "| Test_CvT_model_images           | Test Cvt modle with only images input                           |"
	@echo "| Test_FFN_model                  | Test FFN modle with only parameters input                       |"
	@echo "|                                                                                                   |"
	@echo "|[Tools]                                                                                            |"
	@echo "| memory                          | Show memory usages of CPU & GPU                                 |"
	@echo "| heatmap                         | Show grad_cam of model weights                                  |"
	@echo "| model_plot                      | Plot model's structure                                          |"
	@echo "+---------------------------------------------------------------------------------------------------+"

# # 安裝依賴
# install:
# 	pip install -r requirements.txt

# Prepare
Pick_up_datas:
	$(PYTHON) tools/PickUpData.py

# Train
Train_CvT_model:
	$(PYTHON) models/CvT\(Par\).py

Train_CvT_model_images:
	$(PYTHON) models/CvT\(Img\).py

Train_FFN_model:
	$(PYTHON) models/FFN\(OnlyPar\).py

# Test
Test_CvT_model:
	$(PYTHON) models/CvT_test\(Par\).py

Test_CvT_model_images:
	$(PYTHON) models/CvT_test\(Img\).py

Test_FFN_model:
	$(PYTHON) models/FFN_test\(OnlyPar\).py

#Tools
memory:
	$(PYTHON) tools/memory.py

heatmap:
	$(PYTHON) tools/grad_cam.py

model_plot:
	$(PYTHON) tools/model_plot.py

# 清理編譯生成的文件
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# 偽目標
.PHONY: help Pick_up_datas Train_CvT_model Train_FFN_model Test_CvT_model Test_FFN_model clean

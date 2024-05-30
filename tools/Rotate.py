from PIL import Image
import os

def rotate_and_save(image_path, output_folder, base_filename, start_index):
    # 打開圖片
    original_image = Image.open(image_path)

    # 顯示原始圖片的檔名
    print(f"處理圖片: {image_path}")
    name = start_index
    # 旋轉並儲存圖片
    for angle in [90, 180, 270]:
        rotated_image = original_image.rotate(angle)

        # 產生新的檔名
        new_filename = f'{output_folder}/{base_filename}_{name:02d}.jpg'

        # 顯示旋轉的步驟
        print(f"旋轉 {angle} 度，儲存為: {new_filename}")

        rotated_image.save(new_filename)

        # 更新編號
        name += 200

if __name__ == "__main__":
    # 輸入圖片資料夾路徑
    input_folder = "data/"  # 替換成你的圖片資料夾路徑

    # 如果資料夾不存在，則建立
    if not os.path.exists(input_folder):
        print(f"資料夾 {input_folder} 不存在，請檢查路徑。")
    else:
        # start_index = 201  # 修改起始編號

        # 讀取並處理每張圖片
        for group in range(1, 11):
            for image_num in range(1, 5):
                folder_name = f'trail{group:01d}_{image_num:02d}'
                folder_path = os.path.join(input_folder, f'circle(340x345)/{folder_name}')

                for i in range(200):
                    filename = f'{folder_path}/layer_{i + 1:02d}.jpg'

                    # 呼叫函式執行旋轉並儲存
                    rotate_and_save(filename, folder_path, 'layer', i+201)

                    # start_index += 1

        print("所有圖片旋轉並儲存完成。")

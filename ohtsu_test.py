import numpy as np
import os
import cv2
import glob

def otsu(input_image, min = 0, max = 255):

    #各画素の明るさの集計
    hist = [np.sum(input_image == i) for i in range(256)]

    s_max = (0,-10)

    for th in range(0, 255, 1):

        #閾値で分けたとりあえずのクラスの要素数
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])

        # それぞれのクラスの画素値の平均
        if n1 == 0 : mu1 = 0
        else : mu1 = sum([i * hist[i] for i in range(0,th)]) / n1
        if n2 == 0 : mu2 = 0
        else : mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        # クラス間分散の分子
        s = n1 * n2 * (mu1 - mu2) ** 2

        # クラス間分散の分子の最大値の保持
        if s > s_max[1]:
            s_max = (th, s)

    # クラス間分散が最大のときの閾値を取得
    t = s_max[0]

    # 二値化処理
    input_image[input_image < t] = min
    input_image[input_image >= t] = max

    return input_image


input_image_links = glob.glob("./images/*")

for input_image_link in input_image_links:
    file_name = os.path.splitext(os.path.basename(input_image_link))[0]

    # 入力画像の読み込み
    img = cv2.imread(input_image_link)

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 方法1（NumPyで実装）
    two_value = otsu(gray)

    # 結果を出力
    cv2.imwrite(f"./results/{file_name}_result.png", two_value)
    
    import cv2  #OpenCVのインポート

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #img_binaryを輪郭抽出
    cv2.drawContours(img, contours, -1, (0,0,255), 2) #抽出した輪郭を赤色でimg_colorに重ね書き
    print(len(contours)) #抽出した輪郭の個数を表示する

    cv2.imshow("contours",img) #別ウィンドウを開き(ウィンドウ名 "contours")オブジェクトimg_colorを表示

    cv2.waitKey(0) #キー入力待ち
    cv2.destroyAllWindows() #ウインドウを閉じる
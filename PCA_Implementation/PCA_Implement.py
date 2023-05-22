import numpy as np
import os
import PIL.Image as Image
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt

# file_pathname = r"C:\Users\wxy\YOLOCODE\PCA\2022.9.27\cell0109.png"
# X_reduction = []
# def pca_01(file_pathname):
# for filename in file_pathname:
# file_pathname = "C:\\Users\\wxy\\YOLOCODE\\PCA\\surface-defects"
# a = 0
# def pca_compre(file_pathname):
a = 2065
file_pathname = "C:\\Users\\wxy\\YOLOCODE\\PCA\\defects_additional"
for filename in os.listdir(file_pathname):
        # for i in range(2624):
    img_path = os.path.join(file_pathname, filename)
    img = cv2.imread(img_path)
    img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#cv2彩色图转灰度图的方法！！！
    pca = PCA(n_components='mle', svd_solver='full')
    X_reduction = pca.fit_transform(img_Gray)
    X_reduction[:, 0] = 0
    # X_reduction[:, 1] = 0
    X_generate = pca.inverse_transform(X_reduction)
    # print(X_generate.shape)
    new_img = Image.fromarray(X_generate)
    new_img = new_img.convert('RGB')
    if a <= 2624:
        new_img.save("C:\\Users\\wxy\\YOLOCODE\\PCA\\defects_generate\\%d.png"%(a))
        a += 1
        # break


            # new_img = cv2.cvtColor(X_generate, cv2.COLOR_GRAY2BGR)
            # print(new_img.shape)
# file_pathname = "C:\\Users\\wxy\\YOLOCODE\\PCA\\surface-defects"
# pca_compre("C:\\Users\\wxy\\YOLOCODE\\PCA\\surface-defects")
# plt.imshow(X_generate, cmap=plt.cm.gray, interpolation="nearest")
# new_img.show()
# cv2.imwrite("C:\\Users\\wxy\\YOLOCODE\\PCA\\defects_generate\\1.png", new_img)
# img_new = cv2.cvtColor(X_generate, cv2.COLOR_GRAY2BGR)
# new_img = cv2.imshow('asdadsafsa', new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# print(X_generate.shape)
# print(X_reduction)
# print(pca.explained_variance_ratio_)



# print(img_Gray.shape)






    # image_dir = r"C:\Users\wxy\YOLOCODE\PCA\2022.9.27\cell0109.png"
    # img = Image.open(image_dir)
    # img.show()
    # X = np.asarray(img)

#     return pca_01()
    # pca = PCA(n_components='mle', svd_solver='full')
    # X_reduction = pca.fit_transform(X)
    # print(X_reduction.shape)
    # Y = np.asarray(X_reduction)
    # print(Y.shape)

    # X0 = X_reduction[0]
    # X1 = X_reduction[1]
    # X0_0 = np.zeros_like(X0)
    # X0_1 = np.zeros_like(X1)
    # X_reduction[0] = X0_0
    # X_reduction[1] = X0_1
    # return X_reduction
# image_dir = r"C:\Users\wxy\YOLOCODE\PCA\2022.9.27\cell0109.png"


#file_pathname = r"C:\\Users\\wxy\\YOLOCODE\\PCA\\2022.9.27"
# pca_01(image_dir)
# data = X_reduction
# print(data)


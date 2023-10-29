import cv2
import numpy as np

img = cv2.imread("erp.png")
[src_height, src_width, _] = img.shape

vFOV = 170
hFOV = 170

# theta, phi 가 카메라 pose 관련
theta = 0 # pan
phi = -90 # tilt

# 수업 자료에 나온 것과 다른 f 구하는 방식
# (w/2) * (1/tan(fov/2))
fx = (src_width/2) * 1 / np.tan(np.deg2rad(vFOV/2))
fy = (src_height/2) * 1 / np.tan(np.deg2rad(hFOV/2))

cx = (src_width - 1) / 2.0
cy = (src_height - 1) / 2.0

# camera matrix
K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0,  1],
            ], np.float32)

# w,h 길이 만큼의 배열생성
x = np.arange(src_width)
y = np.arange(src_height)

# x 는 0 ~ w-1 값이 들어간 배열이 w개 있는 행렬
# y 는 각각 0 ~ h-1 의 값만 가진 배열이 h개 있는 행렬
x, y = np.meshgrid(x, y)

# x 와 같은 크기의 1로 구성된 배열 생성
# normalized coordinate 의 z축 값 나타내는듯
z = np.ones_like(x)

# erp 이미지의 normalized coordinate 배열
# 월드좌표
xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

# 방향 벡터
y_axis = np.array([0.0, 1.0, 0.0], np.float32)
x_axis = np.array([1.0, 0.0, 0.0], np.float32)

# 회전축 y에 대한 회전(pan)을 회전변환 행렬로
R1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
# pan 한 회전축 x에 대한 회전(tilt)을 회전변환 행렬로
R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(phi))

R = np.matmul(R2, R1)


# 카메라 pose 적용한 좌표 구하기 ( x = K[R|t]X )
# 역행렬
# rotation matrix 는 직교행렬 이므로 전치행렬이 역행렬
K_inv = np.array([
                [1/fx, 0, -cx/fx],
                [0, 1/fy, -cy/fy],
                [0, 0,  1],
            ], np.float32)

R_inv = R.T

xyz = np.matmul(np.matmul(xyz, K_inv.T), R_inv)

# 위도 경도 구하기
# 벡터 정규화, 계산 편의를 위해서 ?
norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
xyz_norm = xyz / norm

x = xyz_norm[..., 0:1]
y = xyz_norm[..., 1:2]
z = xyz_norm[..., 2:]


# 각각 z축과 x축, z축과 y축으로 평면으로 그려보면 이해가능
lon = np.arctan2(x, z) # 경도
lat = np.arcsin(y) # 위도

# 위도, 경도를 픽셀좌표로 변환
# 경도 -π ~ π, 위도 -π/2 ~ π/2
# 각도 / 2π or π 로 상대적인 위치
# width, height 곱해서 좌표 구함
dst_x = (lon / (2 * np.pi) + 0.5) * (src_width - 1)
dst_y = (lat / np.pi + 0.5) * (src_height - 1)

dst_xy = np.concatenate([dst_x, dst_y], axis=-1).astype(np.float32)

# remapping
dst = cv2.remap(img, dst_xy[..., 0], dst_xy[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


cv2.imshow('test', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite("result.png", dst)
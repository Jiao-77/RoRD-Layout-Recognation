import torch
import torch.nn.functional as F
from models.rord import RoRD
from torchvision import transforms
from utils.transforms import SobelTransform
import numpy as np
import cv2
from PIL import Image

def extract_keypoints_and_descriptors(model, image):
    """
    从 RoRD 模型中提取关键点和描述子。

    参数：
        model (RoRD): RoRD 模型。
        image (torch.Tensor): 输入图像张量，形状为 [1, 1, H, W]。

    返回：
        tuple: (keypoints_input, descriptors)
            - keypoints_input: [N, 2] float tensor，关键点在输入图像中的坐标。
            - descriptors: [N, 128] float tensor，L2 归一化的描述子。
    """
    with torch.no_grad():
        detection_map, _, desc_rord = model(image)
        desc = desc_rord  # 使用 RoRD 描述子头

        # 从检测图中提取关键点
        thresh = 0.5
        binary_map = (detection_map > thresh).float()
        coords = torch.nonzero(binary_map[0, 0] > thresh).float()  # [N, 2]，每个行是 (i_d, j_d)
        keypoints_input = coords * 16.0  # 将特征图坐标映射到输入图像坐标（stride=16）

        # 从描述子图中提取描述子
        # detection_map 的形状为 [1, 1, H/16, W/16]，desc 的形状为 [1, 128, H/8, W/8]
        # 将 detection_map 的坐标映射到 desc 的坐标：(i_d * 2, j_d * 2)
        keypoints_desc = (coords * 2).long()  # [N, 2]，整数坐标
        H_desc, W_desc = desc.shape[2], desc.shape[3]
        mask = (keypoints_desc[:, 0] < H_desc) & (keypoints_desc[:, 1] < W_desc)
        keypoints_desc = keypoints_desc[mask]
        keypoints_input = keypoints_input[mask]

        # 提取描述子
        descriptors = desc[0, :, keypoints_desc[:, 0], keypoints_desc[:, 1]].T  # [N, 128]

        # L2 归一化描述子
        descriptors = F.normalize(descriptors, p=2, dim=1)

        return keypoints_input, descriptors

def mutual_nearest_neighbor(template_descs, layout_descs):
    """
    使用互最近邻（MNN）找到模板和版图之间的匹配。

    参数：
        template_descs (torch.Tensor): 模板描述子，形状为 [M, 128]。
        layout_descs (torch.Tensor): 版图描述子，形状为 [N, 128]。

    返回：
        list: [(i_template, i_layout)]，互最近邻匹配对的列表。
    """
    M, N = template_descs.size(0), layout_descs.size(0)
    if M == 0 or N == 0:
        return []
    similarity_matrix = template_descs @ layout_descs.T  # [M, N]，点积矩阵

    # 找到每个模板描述子的最近邻
    nn_template_to_layout = torch.argmax(similarity_matrix, dim=1)  # [M]

    # 找到每个版图描述子的最近邻
    nn_layout_to_template = torch.argmax(similarity_matrix, dim=0)  # [N]

    # 找到互最近邻
    mutual_matches = []
    for i in range(M):
        j = nn_template_to_layout[i]
        if nn_layout_to_template[j] == i:
            mutual_matches.append((i.item(), j.item()))

    return mutual_matches

def ransac_filter(matches, template_kps, layout_kps):
    """
    使用 RANSAC 对匹配进行几何验证，并返回内点。

    参数：
        matches (list): [(i_template, i_layout)]，匹配对列表。
        template_kps (torch.Tensor): 模板关键点，形状为 [M, 2]。
        layout_kps (torch.Tensor): 版图关键点，形状为 [N, 2]。

    返回：
        tuple: (inlier_matches, num_inliers)
            - inlier_matches: [(i_template, i_layout)]，内点匹配对。
            - num_inliers: int，内点数量。
    """
    src_pts = np.array([template_kps[i].cpu().numpy() for i, _ in matches])
    dst_pts = np.array([layout_kps[j].cpu().numpy() for _, j in matches])

    if len(src_pts) < 4:
        return [], 0

    try:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        if H is None:
            return [], 0
        inliers = mask.ravel() > 0
        num_inliers = np.sum(inliers)
        inlier_matches = [matches[k] for k in range(len(matches)) if inliers[k]]
        return inlier_matches, num_inliers
    except cv2.error:
        return [], 0

def match_template_to_layout(model, layout_image, template_image):
    """
    使用 RoRD 模型执行模板匹配，迭代找到所有匹配并屏蔽已匹配区域。

    参数：
        model (RoRD): RoRD 模型。
        layout_image (torch.Tensor): 版图图像张量，形状为 [1, 1, H_layout, W_layout]。
        template_image (torch.Tensor): 模板图像张量，形状为 [1, 1, H_template, W_template]。

    返回：
        list: [{'x': x_min, 'y': y_min, 'width': w, 'height': h}]，所有检测到的边框。
    """
    # 提取版图和模板的关键点和描述子
    layout_kps, layout_descs = extract_keypoints_and_descriptors(model, layout_image)
    template_kps, template_descs = extract_keypoints_and_descriptors(model, template_image)

    # 初始化活动版图关键点掩码
    active_layout = torch.ones(len(layout_kps), dtype=bool)

    bboxes = []
    while True:
        # 获取当前活动的版图关键点和描述子
        current_layout_kps = layout_kps[active_layout]
        current_layout_descs = layout_descs[active_layout]

        if len(current_layout_descs) == 0:
            break

        # MNN 匹配
        matches = mutual_nearest_neighbor(template_descs, current_layout_descs)

        if len(matches) == 0:
            break

        # 将当前版图索引映射回原始版图索引
        active_indices = torch.nonzero(active_layout).squeeze(1)
        matches_original = [(i_template, active_indices[i_layout].item()) for i_template, i_layout in matches]

        # RANSAC 过滤
        inlier_matches, num_inliers = ransac_filter(matches_original, template_kps, layout_kps)

        if num_inliers > 10:  # 设置内点阈值
            # 获取内点在版图中的关键点
            inlier_layout_kps = [layout_kps[j].cpu().numpy() for _, j in inlier_matches]
            inlier_layout_kps = np.array(inlier_layout_kps)

            # 计算边框
            x_min = int(inlier_layout_kps[:, 0].min())
            y_min = int(inlier_layout_kps[:, 1].min())
            x_max = int(inlier_layout_kps[:, 0].max())
            y_max = int(inlier_layout_kps[:, 1].max())
            bboxes.append({'x': x_min, 'y': y_min, 'width': x_max - x_min, 'height': y_max - y_min})

            # 屏蔽内点
            for _, j in inlier_matches:
                active_layout[j] = False
        else:
            break

    return bboxes

if __name__ == "__main__":
    # 设置变换
    transform = transforms.Compose([
        SobelTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载模型
    model = RoRD().cuda()
    model.load_state_dict(torch.load('path/to/weights.pth'))
    model.eval()

    # 加载版图和模板图像
    layout_image = Image.open('path/to/layout.png').convert('L')
    layout_tensor = transform(layout_image).unsqueeze(0).cuda()

    template_image = Image.open('path/to/template.png').convert('L')
    template_tensor = transform(template_image).unsqueeze(0).cuda()

    # 执行匹配
    detected_bboxes = match_template_to_layout(model, layout_tensor, template_tensor)

    # 打印检测到的边框
    print("检测到的边框：")
    for bbox in detected_bboxes:
        print(bbox)
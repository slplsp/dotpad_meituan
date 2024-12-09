import base64
from myproject.sentence import beam_search_decoder
import cv2
import numpy as np
from django.shortcuts import render, get_object_or_404, redirect
from .models import ImageUpload
from .forms import ImageUploadForm
from ultralytics import YOLO  # YOLOv8 framework
from django.http import JsonResponse


def detect_and_display_edges(image_path):
    # 固定 YOLO 模型路径和填充比例
    yolo_model_path = "yolov8x.pt"  # 请确保该路径正确，指向 YOLOv8 权重文件
    padding = 0.05  # 固定 5% 的填充比例

    # 加载 YOLO 模型
    model = YOLO(yolo_model_path)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return None, None, None

    # 使用 YOLOv8 检测物体
    results = model(image)
    detections = results[0].boxes.xyxy.cpu()
    if len(detections) == 0:
        print("No objects detected.")
        return None, None, None

    # 找到检测到的最大物体区域
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in detections]
    largest_index = np.argmax(areas)
    x_min, y_min, x_max, y_max = map(int, detections[largest_index])

    # 对检测框应用填充
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, x_min - int(padding * width))
    y_min = max(0, y_min - int(padding * height))
    x_max = min(image.shape[1], x_max + int(padding * width))
    y_max = min(image.shape[0], y_max + int(padding * height))

    # 裁剪最大物体并应用边缘检测
    cropped_object = image[y_min:y_max, x_min:x_max]
    bordered_image, edges, binary_array = edge_detection_with_border(cropped_object)

    # 返回三个图像
    return cropped_object, edges, bordered_image, binary_array


def edge_detection_with_border(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用直方图均衡增强对比度
    equalized = cv2.equalizeHist(gray)

    # 使用双边滤波平滑图像
    blurred = cv2.bilateralFilter(equalized, d=9, sigmaColor=75, sigmaSpace=75)

    # 使用 Canny 进行边缘检测
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # 过滤大的轮廓以减少噪声
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_edges = np.zeros_like(edges)
    for contour in contours:
        if cv2.contourArea(contour) > 1:  # 仅保留大的轮廓
            cv2.drawContours(filtered_edges, [contour], -1, 255, thickness=cv2.FILLED)

    # 保持长宽比调整大小为目标尺寸 40x60
    height, width = filtered_edges.shape
    target_width, target_height = 60, 40
    aspect_ratio = width / height
    if aspect_ratio > (target_width / target_height):
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    resized_edges = cv2.resize(filtered_edges, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 创建 40x60 的白色背景
    white_image = np.ones((target_height, target_width), dtype=np.uint8) * 255

    # 将调整大小后的图像居中放置在白色背景上
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    white_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = np.where(resized_edges > 0, 0, 255)

    # 在图像周围添加黑色边框（42x62）
    bordered_image = cv2.copyMakeBorder(white_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    # 生成 40x60 的黑白数组，1 代表黑色，0 代表白色
    binary_array = (white_image == 0).astype(int)

    result = []
    for row in binary_array:
        # 将每行的元素用逗号分隔，并用 [] 包裹
        result.append("[" + ",".join(map(str, row)) + "]")

    # 添加双括号并加入行之间的逗号
    formatted_result = "[" + ",".join(result) + "]"

    return cv2.cvtColor(bordered_image, cv2.COLOR_GRAY2BGR), edges, formatted_result


def result_view(request):
    # 从 session 中获取保存的上下文数据
    context = request.session.get('context', None)
    if not context:
        return redirect('image_to_text_view')  # 如果没有数据则重定向回上传页面
    return render(request, 'image2text/result.html', context)


def image_to_text_view(request):
    uploaded_images = ImageUpload.objects.all()
    form = ImageUploadForm()

    if request.method == 'POST':
        selected_image_id = request.POST.get('selected_image')
        if selected_image_id:
            image_instance = get_object_or_404(ImageUpload, id=selected_image_id)
            image_path = image_instance.image.path
        elif 'image' in request.FILES:
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                image_instance = form.save()
                image_path = image_instance.image.path
        else:
            image_path = None

        if image_path:
            sentence = beam_search_decoder(image_path)
            original_image, edges, edge_image, binary_array = detect_and_display_edges(image_path)

            # 将图像转换为Base64
            def convert_to_base64(img):
                _, buffer = cv2.imencode('.png', img)
                return base64.b64encode(buffer).decode('utf-8')

            # 通过 session 保存 binary_array，以便后续访问
            request.session['binary_array'] = binary_array
            context = {
                'sentence': sentence,
                'original_image': convert_to_base64(original_image),
                'edges_image': convert_to_base64(edges),
                'edge_image': convert_to_base64(edge_image),
            }

            # 使用 `redirect` 跳转到 `result_view`，传递上下文数据
            request.session['context'] = context  # 使用 session 保存数据
            return redirect('result_view')

    return render(request, 'image2text/upload.html', {
        'form': form,
        'uploaded_images': uploaded_images,
    })



def get_2d_array(request):
    # 尝试从 session 中获取保存的 binary_array
    binary_array = request.session.get('binary_array', None)

    # 如果 binary_array 存在并且是字符串形式的二维数组，解析为 Python 数据结构
    if binary_array:
        try:
            # 将字符串形式的 binary_array 转换为二维数组
            array_2d = eval(binary_array)  # 注意：仅在可信环境下使用 eval，否则可能会有安全风险
            return JsonResponse({'array': array_2d})
        except Exception as e:
            return JsonResponse({'error': f'Invalid binary array format: {str(e)}'}, status=400)

    # 如果 binary_array 不存在，返回默认的示例二维数组
    array_2d = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # 添加其他行 ...
               ]
    return JsonResponse({'array': array_2d})

'''
def get_2d_array(request):
    # 示例二维数组
    array_2d = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
[0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0],
[0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,0],
[0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1,1,1,0],
[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0],
[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
[0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
[0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
[0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
[0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0],
[0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0],
[0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    return JsonResponse({'array': array_2d})
'''

def result_view(request):
    # 从 session 中获取保存的上下文数据
    context = request.session.get('context', None)
    if not context:
        return redirect('image_to_text_view')  # 如果没有数据则重定向回上传页面
    return render(request, 'image2text/result.html', context)


def home_view(request):
    return render(request, 'image2text/home.html')


def delete_images_view(request):
    if request.method == 'POST':
        # 获取选中的图片 ID 列表
        image_ids = request.POST.getlist('images_to_delete')
        for image_id in image_ids:
            image = get_object_or_404(ImageUpload, id=image_id)
            image.image.delete()  # 删除实际图片文件
            image.delete()  # 从数据库中删除图片记录
        return redirect('image_to_text_view')  # 删除后返回上传页面

    # GET 请求时显示所有已上传的图片
    uploaded_images = ImageUpload.objects.all()
    return render(request, 'image2text/delete_images.html', {'uploaded_images': uploaded_images})

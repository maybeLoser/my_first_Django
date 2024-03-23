import math

import numpy as np
from app1 import models
from app1.models import Department
from app1.models import UserInfo
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render, redirect

import argparse
import time
# Create your views here.
def depart_list(request):
    "部门列表"
    # 对象
    queryset = Department.objects.all()
    for item in queryset:
        print(item.id, item.title)
    # Department.objects.all().delete()
    # 去数据库中获取所有的部门列表
    return render(request, "depart_list.html", {"queryest": queryset})


def depart_add(request):
    "添加部门"
    # 对象
    if request.method == "GET":
        return render(request, "depart_add.html")
    # 获取用户提交的数据(title输入为空怎么办)
    title = request.POST.get("title")
    if title:  # Check if the title is not empty
        # 添加到数据库
        Department.objects.create(title=title)
        print(title)
        # 添加完成后可以返回一个添加成功，也可像下面一样优化
        # return HttpResponse("添加成功")

        # 自动跳转（自己的网站也可以写成/info/list/）
        return HttpResponseRedirect("http://127.0.0.1:8000/depart/list/")


def depart_delete(request):
    "删除部门"
    nid = request.GET.get("nid")
    Department.objects.filter(id=nid).delete()
    # 自动跳转（自己的网站也可以写成/info/list/）
    return HttpResponseRedirect("http://127.0.0.1:8000/depart/list/")


def depart_edit(request, nid):
    "修改部门"
    if request.method == "GET":
        # 根据nid获取数据
        row_object = Department.objects.filter(id=nid).first()
        return render(request, "depart_edit.html", {"row_object": row_object})

    # 获取用户提交的标题
    title = request.POST.get("title")

    # 根据ID找到数据库中数据并且更新
    Department.objects.filter(id=nid).update(title=title)

    # 自动跳转（自己的网站也可以写成/info/list/）
    return HttpResponseRedirect("http://127.0.0.1:8000/depart/list/")


def user_list(request):
    "用户管理"
    queryest = UserInfo.objects.all()
    for obj in queryest:
        # datatime类型转化成字符串 res = dt.strftime("%Y-%m-%d"-%H-%M)
        print(obj.id, obj.name, obj.account, obj.create_time.strftime("%Y-%m-%d"), obj.gender, obj.get_gender_display(),
              obj.depart.title)
        # obj.gender #1,2
        # obj.get_gender_display() #get_字段名称_display();
        # obj.depart_id #获取数据库中的元素原始值
        # obj.depart.title #根据id自动关联的表中那一行数据的depart对象
    return render(request, "user_list.html", {"queryest": queryest})


def user_add(request):
    "添加用户"
    if request.method == "GET":
        context = {
            "gender_choices": models.UserInfo.gender_choices,
            "depart_list": models.Department.objects.all()
        }
        return render(request, "user_add.html", context)

    # 获取用户提交数据
    name = request.POST.get("name")
    password = request.POST.get("password")
    age = request.POST.get("age")
    account = request.POST.get("account")
    create_time = request.POST.get("create_time")
    gender = request.POST.get("gender")
    depart = request.POST.get("depart")

    # 数据校验

    # 添加数据到数据库中
    models.UserInfo.objects.create(name=name, password=password, age=age, account=account, create_time=create_time,
                                   gender=gender, depart=depart)

    # 返回到用户页面
    return HttpResponseRedirect("http://127.0.0.1:8000/user/list/")


from django import forms


class UserModelForm(forms.ModelForm):
    # 更多校验功能
    name = forms.CharField(min_length=2, label="姓名")
    password = forms.CharField(min_length=8, label="密码")

    class Meta:
        model = models.UserInfo
        fields = ["name", "password", "age", "account", "create_time", "gender", "depart"]

        # 复杂写法
        # widgets = {
        #     "name":forms.TextInput(attrs={"class":"form-control"}),
        #     "password": forms.PasswordInput(attrs={"class": "form-control"})
        # }

        # 简单写法

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 找到所有插件添加样式
        for name, field in self.fields.items():
            # 如有需要跳过写以下条件
            if name == "name":
                continue
            field.widget.attrs = {"class": "form-control", "placeholder": field.label}


def user_model_form_add(request):
    "添加用户ModelForm版本"
    if request.method == "GET":
        form = UserModelForm()
        return render(request, "user_model_form_add.html", {"form": form})

    # POST提交数据，数据校验.
    form = UserModelForm(data=request.POST)
    # 自动校验,如果数据合法，保存到数据库
    if form.is_valid():
        # 获取提交数据
        # print(form.cleaned_data)
        # form中的自动保存功能
        form.save()
        return redirect("/user/list/")
    # 校验失败，在页面上显示信息，form中封装了每一个字段的错误信息
    # print(form.errors)
    return render(request, "user_model_form_add.html", {"form": form})


def user_edit(request, nid):
    "编辑用户"
    row_object = models.UserInfo.objects.filter(id=nid).first()
    if request.method == "GET":
        # 根据ID去数据库获取需要编辑的哪一行数据，instance默认填充
        form = UserModelForm(instance=row_object)
        return render(request, "user_edit.html", {"form": form})

    form = UserModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        # 默认保存的时用户输入的所有数据，如果想在用户输入外在添加一段值
        # form.instance.字段值 = 值
        form.save()
        return redirect("/user/list/")
    return render(request, "user_edit.html", {"form": form})


def user_delete(request, nid):
    "删除部门"
    models.UserInfo.objects.filter(id=nid).delete()
    # 自动跳转（自己的网站也可以写成/user/list/）
    return redirect("http://127.0.0.1:8000/user/list/")

def pretty_list(request):
    "靓号列表"
    data_dict = {}
    #通过URL获取数据
    value = request.GET.get("q")
    if value:
     # 筛选功能
            data_dict = {
                ["mobile__contains"]: value
            }

    #order_by排序功能 order_by("level")降序-级别低的在上面
    #相当于MySQL中select * from 表 order_by level asc;
    #order_by排序功能 order_by("-level")降序-级别高的在上面
    #相当于MySQL中select * from 表 order_by level dasc;
    queryest = models.PrettyNum.objects.filter(**data_dict).order_by("-level")

        # obj.gender #1,2
        # obj.get_gender_display() #get_字段名称_display();
        # obj.depart_id #获取数据库中的元素原始值
        # obj.depart.title #根据id自动关联的表中那一行数据的depart对象
    return render(request, "pretty_list.html", {"queryest": queryest})


from django.core.exceptions import ValidationError
class PrettyModelForm(forms.ModelForm):

    #第一种限制手机号格式的方法
    # mobile = forms.CharField(min_length=2,
    #                          label="手机号",
    #                          #手机正则设置
    #                          validators=[RegexValidator(r'^1[3~9]\d{9}$',"手机号格式错误")]
    #                          )
    class Meta:
        model = models.PrettyNum
        fields = ["mobile","price", "level", "status"]
        # 自建 fields = ["mobile", "password", "price", "level", "status"]
        # 所有字段 fields = "__all__"
        # 排除字段 exclude = [ "level" ]



    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 找到所有插件添加样式
        for name, field in self.fields.items():
            field.widget.attrs = {"class": "form-control", "placeholder": field.label}

    #限制方式2
    def clean_mobile(self):
        txt_mobile = self.cleaned_data["mobile"]

        exists = models.PrettyNum.objects.filter(mobile=txt_mobile).exists()
        if exists:
            raise ValidationError("手机号已经存在")

        if len(txt_mobile) != 11:
            raise ValidationError("格式错误")
        # 验证通过，返回用户输入的值
        return txt_mobile


def pretty_add(request):
    "添加靓号"
    if request.method == "GET":
        form = PrettyModelForm()
        return render(request, "pretty_add.html",{"form":form})
    # POST提交数据，数据校验.
    form = PrettyModelForm(data=request.POST)
    # 自动校验,如果数据合法，保存到数据库
    if form.is_valid():
        # 获取提交数据
        # print(form.cleaned_data)
        # form中的自动保存功能
        form.save()
        return redirect("/pretty/list/")
    # 校验失败，在页面上显示信息，form中封装了每一个字段的错误信息
    # print(form.errors)
    return render(request, "pretty_add.html", {"form": form})

#编辑专用modelform，只允许用户修改价格级别状态
class PrettyEditModelForm(forms.ModelForm):
    class Meta:
        model = models.PrettyNum
        fields = ["mobile","price", "level", "status"]
        # 自建 fields = ["mobile", "password", "price", "level", "status"]
        # 所有字段 fields = "__all__"
        # 排除字段 exclude = [ "level" ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 找到所有插件添加样式
        for name, field in self.fields.items():
            field.widget.attrs = {"class": "form-control", "placeholder": field.label}

    #若在编辑数据时，需要排除自己以外的账号
    def clean_mobile(self):

        #当前编辑的一行ID
        #print(self.instance.pk)

        txt_mobile = self.cleaned_data["mobile"]
        #判断号码是否存在且不为自己输入的号码
        exists = models.PrettyNum.objects.exclude(id=self.instance.pk).filter(mobile = txt_mobile).exists()
        if exists:
            raise ValidationError("手机号已经存在")

        if len(txt_mobile)!= 11:
            raise ValidationError("格式错误")
        #验证通过，返回用户输入的值
        return txt_mobile
def pretty_edit(request, nid):
    "编辑靓号"
    row_object = models.PrettyNum.objects.filter(id=nid).first()
    if request.method == "GET":
        # 根据ID去数据库获取需要编辑的哪一行数据，instance默认填充
        form = PrettyEditModelForm(instance=row_object)
        return render(request, "pretty_edit.html", {"form": form})

    form = UserModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        # 默认保存的时用户输入的所有数据，如果想在用户输入外在添加一段值
        # form.instance.字段值 = 值
        form.save()
        return redirect("/pretty/list/")
    return render(request, "pretty_edit.html", {"form": form})

def pretty_delete(request, nid):
    "删除部门"
    models.PrettyNum.objects.filter(id=nid).delete()
    # 自动跳转（自己的网站也可以写成/user/list/）
    return redirect("http://127.0.0.1:8000/pretty/list/")


from django.core.files.storage import FileSystemStorage
from PIL import Image, ImageEnhance
import random
import os
import cv2
# upload_image view function
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']

        # 处理图像
        original_image = Image.open(image)
        contrast_factor = random.uniform(0.5, 1.5)
        enhanced_image = ImageEnhance.Contrast(original_image).enhance(contrast_factor)

        # 获取用户输入的保存路径
        output_path = request.POST.get('output_path', '')

        if not output_path:
            # 如果用户未输入保存路径，则默认使用桌面路径
            output_path = os.path.join(os.path.expanduser("~"), 'Desktop')

        # 创建输出文件夹（如果不存在）
        output_folder = os.path.join(output_path, "cropped_images")
        os.makedirs(output_folder, exist_ok=True)

        # 生成输出文件路径
        enhanced_filename = f'enhanced_{image.name}'
        enhanced_image_path = os.path.join(output_folder, enhanced_filename)
        enhanced_image.save(enhanced_image_path)

        context = {
            'original_image_url': image.url,
            'enhanced_image_url': enhanced_image_path,
        }

        return render(request, 'upload.html', context)
    return render(request, 'upload.html')

def image_result(request):
    return render(request, 'result.html')

#各种图像增强功能
#运动模糊
def motion_blur(image, degree, angle=45):
    degree = int(degree)
    image = np.asarray(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    image = Image.fromarray(blurred)
    return image

#高斯模糊
def ssuasion_blur(image):

    blur_size = (5, 5)  # 高斯核大小，必须为奇数
    # sigma = float(sigma) # 高斯核标准差
    sigma=2.5
    # 读取RGB格式的图片
    img = cv2.imread(image)

    # 将RGB格式的图像转换为BGR格式的图像
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 显示转换后的图像

    blurred_image = cv2.GaussianBlur(bgr_img, blur_size, sigma)


    return Image.fromarray(blurred_image)

#高斯噪声
def ssuasion_noise(image, mean, var):
    """
        添加高斯噪声
        mean : 均值
        var : 方差
    """
    img_array=np.asarray(image)
    imgarray = img_array/255.0
    img_bytes = image.tobytes()
    img = Image.frombytes(image.mode,image.size,img_bytes)
    imgarra = np.array(imgarray,dtype=np.float32)
    mean=float(mean)
    var=float(var)
    noise = np.random.normal(mean, var ** 0.5, imgarra .shape)
    out = imgarra + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out

#随机擦除
def randomErasing(image, sl=0.02, sh=0.4, r1=0.3):
    img = cv2.imread(image)
    area = img.shape[0] * img.shape[1]

    target_area = random.uniform(sl, sh) * area
    aspect_ratio = random.uniform(r1, 1 / r1)

    h = int(round(math.sqrt(target_area * aspect_ratio)))
    w = int(round(math.sqrt(target_area / aspect_ratio)))

    if w < img.shape[1] and h < img.shape[0]:
        x1 = random.randint(0, img.shape[0] - h)
        y1 = random.randint(0, img.shape[1] - w)

        img[x1:x1 + h, y1:y1 + w, :] = 0  # 将指定区域的像素值设置为0（黑色）

    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return image

#随机擦除
def random_contrast(image, lower=0.5, upper=1.5):
    contrast_factor = np.random.uniform(lower, upper)
    img = image.astype(np.float32)
    img = img * contrast_factor
    out = np.clip(img, 0, 255).astype(np.uint8)
    return out

#动画效果
def animated(image_path, num_frame):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像从BGR转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # 设置动画帧数和过渡时间
    num_frames = int(num_frame)
    transition_time = 1.0 / num_frames

    # 创建动画
    frames = []
    for i in range(num_frames):
        alpha = i * transition_time

        # 对图像进行编辑，例如增加亮度
        edited_image = image + alpha * 50  # 增加亮度

        # 将结果图像保存到帧列表中
        frames.append(edited_image)

    return frames

#增强边缘
def enhance_edges(img,blurKsize = 7,edgeKsize = 5):
    if blurKsize >=3:
        blurredSrc = cv2.medianBlur(img,blurKsize)
        graySrc = cv2.cvtColor(blurredSrc,cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc,cv2.CV_8U,graySrc,ksize = edgeKsize )
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(img)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    dst = cv2.merge(channels)
    return dst

#透视变化
def Perspective_Transformed(image):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file")
    ap.add_argument("-c", "--coords",help="comma seperated list of source points")
    args = vars(ap.parse_args())
    image = cv2.imread(args[image])
    pts = np.array(eval(args["coords"]), dtype="float32")
    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    warped = four_point_transform(image, pts)
    return warped

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

#HSV变换
def hsv(image):
    # 读取图像
    # 将图像从 RGB 颜色空间转换为 HSV 颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 在 HSV 颜色空间中调整色调、饱和度和亮度参数
    hue_shift = 30  # 色调偏移量，范围为 -180 到 180
    saturation_scale = 1.5  # 饱和度缩放因子
    value_scale = 1.2  # 亮度缩放因子
    # 分离 HSV 通道
    h, s, v = cv2.split(hsv_image)
    # 对色调（H）通道进行平移
    h = (h + hue_shift) % 180
    # 对饱和度（S）和亮度（V）通道进行缩放
    s = cv2.multiply(s, saturation_scale)
    v = cv2.multiply(v, value_scale)
    # 合并 HSV 通道
    hsv_image_enhanced = cv2.merge((h, s, v))
    # 将图像从 HSV 颜色空间转换回 RGB 颜色空间
    rgb_image_enhanced = cv2.cvtColor(hsv_image_enhanced, cv2.COLOR_HSV2BGR)
    return rgb_image_enhanced

#随机亮度
def random_bright(image_path):
    image = Image.open(image_path)
    # 将图像从BGR转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    brightness_range = (0.5,1.5)
    brightness_factor = random.uniform(*brightness_range)
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(brightness_factor)
    return image

def localEqualHist(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(7,7))
    dst = clahe.apply(gray)
    return dst

def SaltAndPepper(img_path,percetage):
    image = cv2.imread(img_path)
    # 将图像从BGR转换为RGB
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    SP_NoiseImg=np.array(image)
    percetage=int(percetage)
    h,w,c=SP_NoiseImg.shape
    print(SP_NoiseImg.shape)
    SP_NoiseNum=int(percetage*h*w*c)
    high = SP_NoiseImg.shape[0] * SP_NoiseImg.shape[1]
    noise_pixels=int(high*percetage/100)
    print(noise_pixels)
    noise_index=np.random.randint(0,high,size=noise_pixels)
    for i in noise_index:
        x = i //SP_NoiseImg.shape[1]
        y = i % SP_NoiseImg.shape[1]
        if np.random.randint(0,1)==0:
            SP_NoiseImg[x,y]=255
        else:
            SP_NoiseImg[x,y]=50
    return SP_NoiseImg


from tkinter import Tk
from tkinter.filedialog import askopenfilename

#渐变蒙版
def blur_image(img_path):
    image = cv2.imread(str(img_path))
    # 创建蒙版
    mask = np.zeros_like(image)
    # 创建渐变蒙版
    height, width, _ = image.shape
    grad_mask = np.zeros((height, width), dtype=np.uint8)
    grad_mask[:, :width//2] = 255
    grad_mask = cv2.GaussianBlur(grad_mask, (15, 15), 0)
    # 将渐变蒙版转换为三通道图像
    grad_mask = cv2.cvtColor(grad_mask, cv2.COLOR_GRAY2BGR)
    #将图像通过渐变蒙版混合
    result = cv2.addWeighted(image, 0.5, mask, 0.5, 0, mask=grad_mask)
    return result
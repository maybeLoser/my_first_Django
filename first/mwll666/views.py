from django.shortcuts import render,HttpResponse,redirect
from django.conf import settings
from django.http import HttpResponse
from django.http import HttpResponseRedirect

# Create your views here.
#给个反应
def index(request):
    return HttpResponse('mwll666666')

#打开HTMl文件
def list(request):
    return render(request, "M1-first.html")

#重定向到某网页
def redirect(request):
    return HttpResponseRedirect('https://limestart.cn/')

#注册页面
def login(request):
    if request.method == "GET":#请求方法    上传GET/提交POST
        return render(request, "login.html")
    else:
        #如果是POST请求，则获取用户提交的数据
        print(request.POST)
        username=request.POST.get("username")
        password = request.POST.get("password")
        if username == "LL" and password == "22":
            return redirect("https://limestart.cn/")
        else:
            #密码错误
            return render(request, "login.html",{"error_msg":"用户名或者密码错误"})



from mwll666.models import Department
from mwll666.models import UserUnfo
def orm(request):
    #测试ORM表中的数据
    Department.objects.create(title="销售部")
    UserUnfo.objects.create(name = "毛大哥",password="111",age= 2000)
    return HttpResponse("实验成功")

def info_list(request):
    #第一步获取数据库中所有用户信息
    data_list=UserUnfo.objects.all()
    print(data_list)
    for item in data_list:
        print(item.id,item.name,item.password,item.age)
    return render(request,"info_list.html",{"data_list":data_list})

def info_add(request):
    if request.method == "GET":
        return render(request, "info_add.html")
   #获取用户提交的数据
    user=request.POST.get("user")
    pwd=request.POST.get("pwd")
    age=request.POST.get("age")

    #添加到数据库
    UserUnfo.objects.create(name = user,password = pwd,age= age)

    #添加完成后可以返回一个添加成功，也可像下面一样优化
    #return HttpResponse("添加成功")

    #自动跳转（自己的网站也可以写成/info/list/）
    return HttpResponseRedirect("http://127.0.0.1:8000/info/list/")


def info_delete(request):
    nid=request.GET.get("nid")
    UserUnfo.objects.filter(id=nid).delete()
    # 自动跳转（自己的网站也可以写成/info/list/）
    return HttpResponseRedirect("http://127.0.0.1:8000/info/list/")




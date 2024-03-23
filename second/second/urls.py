"""
URL configuration for second project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from app1 import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),

    # 部门管理
    path("depart/list/", views.depart_list),
    path("depart/add/", views.depart_add),
    path("depart/delete/", views.depart_delete),
    path("depart/<int:nid>/edit/", views.depart_edit),

    # 用户管理
    path("user/list/", views.user_list),
    path("user/add/", views.user_add),
    path("user/model/form/add/", views.user_model_form_add),
    path("user/<int:nid>/edit/", views.user_edit),
    path("user/<int:nid>/delete/", views.user_delete),

    # 靓号管理
    path("pretty/list/", views.pretty_list),
    path("pretty/add/", views.pretty_add),
    path("pretty/<int:nid>/edit/", views.pretty_edit),
    path("pretty/<int:nid>/delete/", views.pretty_delete),


    # 可视化设置

    path('upload/', views.upload_image, name='upload_image'),
    path('result/', views.image_result, name='image_result'),  # 添加结果页面的URL
]

# 配置处理媒体文件的URL
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
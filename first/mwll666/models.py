from django.db import models

# Create your models here.
# 类
class UserUnfo(models.Model):
    name = models.CharField(max_length=32)
    password = models.CharField(max_length=64)  # 同上为字符串
    age = models.IntegerField()  # 整型


class Department(models.Model):
    title = models.CharField(max_length=32)

#新建数据  （insert into mwll666_Department(title)values())
#Department.objects.create(title="销售部")
#UserUnfo.objects.create(name="111",password ="222",age = "19")
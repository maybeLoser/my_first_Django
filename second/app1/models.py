

from django.conf import settings
from django.db import models
from django.utils import timezone


class Department(models.Model):
    """这是部门表"""
    # verbose_name为注解,BigAutoField指代big int类型,AutoField指代int类型
    # apps中已经自动配置了id
    #id = models.BigAutoField(verbose_name="ID",max_length=True,unique=True)
    title = models.CharField(verbose_name="部门标题", max_length=32)

    def __str__(self):
        return self.title

class UserInfo(models.Model):
    """这是员工表"""
    name = models.CharField(verbose_name="姓名", max_length=16)
    password = models.CharField(verbose_name="密码", max_length=64)
    age = models.IntegerField(verbose_name="年龄")
    # 数字长度是10,小数长度为2,默认为零
    account = models.DecimalField(verbose_name="账户余额", max_digits=10, decimal_places=2, default=0)
    # 时间类型DateTimeField  年月日DateField
    create_time = models.DateField(verbose_name="入职时间")

    # 无约束
    # depart_id= models.BigIntegerField(verbose_name="部门ID")

    # 有约束
    # ~to表示与哪张表关联    to.fields表示与表中的哪一列关联
    # django自动写的depart
    # 生成数据列depart_id
    # 1.部门表被删除 级联删除
    # depart = models.ForeignKey(to="Department", to_fields="id",on_delete = models.CASCADE)
    # 2.部门表被删除 置空
    depart = models.ForeignKey(verbose_name="部门",to="Department",null=True, blank=True, on_delete=models.SET_NULL)

    # 元组 在Django中做的约束
    gender_choices = (
        (1, "男"),
        (2, "女"),
    )
    gender = models.SmallIntegerField(verbose_name="性别", choices=gender_choices)


class PrettyNum(models.Model):
    "靓号表"
    mobile = models.CharField(verbose_name="手机号",max_length=11)
    price = models.IntegerField(verbose_name="价格",default = 0, null = True , blank = True)
    level_choice = {
        (1, "一级"),
        (2, "二级"),
        (3, "三级"),
        (4, "四级"),
    }
    level = models.SmallIntegerField(verbose_name="级别", choices=level_choice, default=1)
    status_choices={
        (1, "已占用"),
        (2, "未使用"),
    }
    status = models.SmallIntegerField(verbose_name="状态", choices=status_choices, default=2)

class ProcessedImage(models.Model):
    original_image = models.ImageField(upload_to='original_images/')
    processed_image = models.ImageField(upload_to='processed_images/')


#博客文章网页
class Post(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    text = models.TextField()
    created_date = models.DateTimeField(default=timezone.now)
    published_date = models.DateTimeField(blank=True, null=True)

    def publish(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.title

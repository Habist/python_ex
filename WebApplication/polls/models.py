import datetime
from django.db import models
from django.utils import timezone
from Deep_Learning.Neural_Network import main as cnn


# Create your models here.


class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)

    def __str__(self):
        return self.question_text


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text


class ConvNeuralNetwork(models.Model):
    train_class = cnn.load_instance('train_class')

    def __str__(self):
        return 'test_ver 1.0'

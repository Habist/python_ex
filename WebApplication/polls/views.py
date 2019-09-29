from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.urls import reverse
from django.views import generic
from django.template import loader
from .models import Choice, Question, ConvNeuralNetwork
import numpy as np
import json

# Create your views here.


class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        """Return the last five published questions."""
        return Question.objects.order_by('-pub_date')[:5]


class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail.html'


class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/results.html'


# def index(request):
#     latest_question_list = Question.objects.order_by('-pub_date')[:5]
#     # template = loader.get_template('polls/index.html')
#     # context = {
#     #     'latest_question_list': latest_question_list,
#     # }
#     # return HttpResponse(template.render(context, request))
#     ### 템플릿에 context를 채워넣어 표현한 결과를 HttpRespose와 같이 반환하는 형태를 단축 기능을 통해 사용가능
#     context = {'latest_question_list': latest_question_list}
#     return render(request, 'polls/index.html', context)

def test(request):
    # idx = 5
    # ConvNeuralNetwork.train_class.predict(idx)
    return render(request, 'polls/test.html', {})


def send(request):
    print('send@@@@@@@@@@')
    arr = json.loads(request.POST['image'])
    np_arr = np.array(arr)
    np_arr = np_arr.astype(np.float32)
    np_arr /= 255.0
    np_arr = np_arr.reshape(1,1,28,28)
    ConvNeuralNetwork.train_class.predict(np_arr)
    context = {
        'status' : 'success'
    }
    return HttpResponse(json.dumps(context), content_type="application/json")


# def detail(request, question_id):
#     # try:
#     #     question = Question.objects.get(pk=question_id)
#     # except Question.DoesNotExist:
#     #     raise Http404("Question does not exist")
#     # return render(request, 'polls/detail.html', {'question': question})
#     ### 단축 기능 get_object_or_404
#     question = get_object_or_404(Question, pk=question_id)
#     return render(request, 'polls/detail.html', {'question': question})


# def results(request, question_id):
#     question = get_object_or_404(Question, pk=question_id)
#     return render(request, 'polls/results.html', {'question': question})


def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
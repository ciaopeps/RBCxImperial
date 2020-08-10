from classes.Preprocessing import Preprocessing
from classes.ClassifierModel import ClassifierModel
from classes.TopicModel import TopicModel

from classes.Ner import Ner
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config
import formencode
import json
import os
import webbrowser
import sys
import pandas as pd
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

params = json.load(open(os.path.join('./configs', 'serviceconfigs.json'), 'r'))
host = params.get('host', 'localhost')
port = int(params.get('port', 5000))


# Define the microservice class
@view_config(route_name="/RBCxImperial")
class Microservice(object):

    def __init__(self, request):
        global configspath
        global modelpath
        self.request = request
        self.cm = ClassifierModel()
        self.ner = Ner()
        self.tc = TopicModel()

    def __call__(self, *args, **kwargs):

        if self.request.method == 'GET':
            page = open('local/Upload.html', 'rb')
            page = BeautifulSoup(page, features="html.parser").prettify(encoding='utf8')
            return Response(page)

        elif self.request.method == 'POST':
            print('log:: POST request received. Starting...')
            try:
                multiDict = self.request.POST
                print('log:: Received dict: {}'.format(multiDict))
                mydict = formencode.variabledecode.variable_decode(multiDict)
                myFile = pd.read_csv(mydict['myFile'].file)
                predictions = self.cm.serve_predictions(myFile)

                output = myFile.drop(
                    columns=['real', 'fake']
                )
                output['labels'] = (predictions)
                real = output[output['labels'] == 0]
                real['Summary'] = real['text'].apply(lambda x: Ner.summary(x))
                real['Key People'] = real['text'].apply(lambda x: Ner.get_people(x))
                real['Key Org'] = real['text'].apply(lambda x: Ner.get_org(x))
                predictionstopic = self.tc.serve_predictions(real)
                real['Topic'] = predictionstopic
                real = real.drop(
                    columns=['subject','clean','labels']
                )
                real = real.rename(columns={"title": "Title", "text": "Article", "date": "Date"})

                return Response(real.to_html(), status=201)

            except Exception as e:
                return Response("An exception has occurred: {}".format(str(e)))
        else:
            return Response("Hey! I do not accept other kind of requests except GET or POST!!!")


if __name__ == '__main__':

    if len(sys.argv) > 1:
        pp = Preprocessing()
        dataset, dataset_sample = pp.extract()
        dataset = pd.read_csv('data/dataset.csv')
        dataset_sample = pd.read_csv('data/dataset_sample.csv')
        cm = ClassifierModel()
        tc = TopicModel()
        ner = Ner()
        cm.fit(dataset)
        tc.run()

    else:

        with Configurator() as config:
            config.add_static_view('main/', './data')
            config.add_view(Microservice, route_name="RBCxImperial")
            config.add_route("RBCxImperial", '/RBCxImperial')
            app = config.make_wsgi_app()

        print('log:: Starting the microservice...')
        print('log:: Host: {} - Port: {}'.format(host, port))

        server = make_server(host, port, app)

        webbrowser.open('http://' + host + ':' + str(port) + '/RBCxImperial', new=2)

        server.serve_forever()

    pass

import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import load_model
from .serializers import AccelerometerSerializer

import numpy as np
from pprint import pprint

from .res.activity_recognition import main

import threading

count = 0
model = load_model("model.h5")


class AccelerometerView(APIView):
    def post(self, request, format=None, *args, **kwargs):
        serializer = AccelerometerSerializer(data=request.data)
        # print(serializer.data)
        if serializer.is_valid():
            data = serializer.data
            # main_thread = threading.Thread(target=_main, args=[data['values']], daemon=True)
            # main_thread.start()
            global count
            count += 1
            print(count)
            _main(data['values'])
            return Response({'a': count}, status=status.HTTP_200_OK)
        return Response({}, status=status.HTTP_400_BAD_REQUEST)


def _main(ls):
    # print(len(ls))
    print('length is', len(ls))
    #sensor_data = [float(ls[i]) for i in range(270)]  # 90(x,y,z)data i.e 270 data
    input_node = []
    for a in range(90):
        b = [[ls[a * 3]], [ls[3 * a + 1]], [ls[3 * a + 2]]]
        input_node.insert(a, b)
    input_node = np.array(input_node)
    pprint(input_node)
    print("\n\n")
    print("START TIME:  ", str(datetime.datetime.now().time()))
    prediction = model.predict(input_node.reshape(-1, 90, 3, 1))
    print("\n\n")
    label = ['downstair', 'jogging', 'sitting', 'standing', 'upstair', 'walking']
    result = label[np.argmax(prediction)]
    print(result)
    print()
    print("END TIME:  ", str(datetime.datetime.now().time()))
    return

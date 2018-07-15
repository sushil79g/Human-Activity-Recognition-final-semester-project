# import numpy as np
#
#
# from .activity_recognition import main
#
#
# def _main(inputdata):
#     model = main()
#     while True:
#         prediction = model.predict(inputdata.reshape(-1, 90, 3, 1))
#         label = ['downstair', 'jogging', 'sitting', 'standing', 'upstair', 'walking']
#         result = label[np.argmax(prediction)]
#         print(result)

import matplotlib.pyplot as plt
from functions import EER_output, FAR_given_FRR_output,plot, roc_curve_draw
import time
import logging

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%d-%m %H:%M',
                        filename='C:/Users/ahmet/PycharmProjects/biometrics/logs.txt',
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)

    logger_EER = logging.getLogger('EER_output')
    logger_FAR_FRR = logging.getLogger('FAR_given_FRR_output')
    logger_plot = logging.getLogger('plot')
    logger_ROC = logging.getLogger('roc_curve_draw')

    start = time.time()

    EER_output(data_p='data1_SM.txt', labels_p='data1_Class_Labels.txt',outputname='EER_output_1')
    logger_EER.info('data1 done.')

    FAR_given_FRR_output(data_p='data1_SM.txt', labels_p='data1_Class_Labels.txt',outputname='FAR_0.1_given_FRR_output_1'
                         ,given_far=0.1)
    logger_FAR_FRR.info('FAR = 0.1 / data1 done.')

    plot(data_p='data1_SM.txt', labels_p='data1_Class_Labels.txt',save_name='plot1.png')
    logger_plot.info('data1 done.')

    roc_curve_draw('data1_SM.txt', 'data1_Class_Labels.txt', color='red', label='data1', plottype='lineplot')
    logger_ROC.info('data1 done.')

    plt.xlabel('FAR')
    plt.ylabel('GAR')

    plt.savefig('ROC_')
    logger_ROC.info('ROC plot saved.')

    end = time.time()

    logging.info('Elapsed time: ' + str(end-start))


import numpy as np
import matplotlib.pyplot as plt
import random

class xject_helper:    
    
    #plota predicao
    #criado por: https://www.tensorflow.org/tutorials/keras/classification
    def plot_prediction(class_names, i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        #plt.xticks([])
        #plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions_array)
        color = 'blue' if predicted_label == true_label else 'orange'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 
                                             100*np.max(predictions_array), class_names[true_label]), color=color)
        
    #plota valor do array
    #criado por: https://www.tensorflow.org/tutorials/keras/classification
    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
        
        
    #plota um conjunto de imagens
    #criado por: falreis
    def plot_images(images, labels, class_names, rows=5, columns=8, binary=False, random=True):
        idx = -1
        
        fig=plt.figure(figsize=(2*columns, 2*rows*1.2))
        plt.axis('off')
        plt.grid(False)

        for i in range(1, columns*rows +1):
            if(random):
                idx = random.randrange(0, len(images))
            else:
                idx += 1
            img = images[idx]

            fig.add_subplot(rows, columns, i)
            
            if(binary):
                ax = plt.imshow(img, cmap=plt.cm.binary)
            else:
                ax = plt.imshow(img)
                
            plt.title(class_names[labels[idx]])
            plt.axis('off')
        plt.show()
        
        
    def plot_pred(class_names, predictions_array, true_labels, images, rows=5, columns=3):
        idx = -1
        columns *=2

        fig=plt.figure(figsize=(3*columns, 2.8*rows*1.5))
        plt.axis('off')
        plt.grid(False)

        for i in range(0, int((columns*rows)/2)):
            predict_arr, data_label, img = predictions_array[i], true_labels[i], images[i]
            
            #imprime imagem
            fig.add_subplot(rows, columns, (2*i)+1)
            plt.imshow(img, cmap=plt.cm.binary)
            plt.axis('off')
            
            #imprime titulo
            predicted_label = np.argmax(predict_arr)
            color = 'black' if predicted_label == data_label else 'red'
            plt.title("Prev: {} ({:2.0f}%) \nTrue: {}".format(
                                                        class_names[predicted_label],
                                                        100*np.max(predict_arr),
                                                        class_names[data_label])
                        , color=color, loc='left', fontsize = 12)
            
            #imprime gráficos
            fig.add_subplot(rows, columns, (2*i)+2)
            #plt.axis('off')
            
            thisplot = plt.barh(range(10), predict_arr, height=0.5, color="#777777")
            plt.xlim([0, 1])
            plt.xlabel('Probabilidade')
            plt.ylabel('Dígito')
            plt.yticks(range(10))
            predicted_label = np.argmax(predict_arr)

            thisplot[predicted_label].set_color('red')
            thisplot[data_label].set_color('blue')
            
        plt.show()
        
                
    def plot_history_training(history, metrics, legend, ylabel=''):
        fig=plt.figure(figsize=(8, 5))
        
        for m in metrics:
            plt.plot(history.history[m])

        #exibe legenda, títulos e nomes dos eixos
        plt.ylabel(ylabel)
        plt.xlabel('Épocas')
        plt.legend(legend, loc='upper left')
        plt.show()

        
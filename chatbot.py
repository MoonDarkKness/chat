#SE HACEN LAS IMPORTACIONES DE LA LIBRERIA
import random #ALEATORIA
import json #POR EL FORMATO
import pickle 
import numpy as np #PARA MATRICES
import nltk #PARA TRABAJAR INTERPRETANDO TEXTO
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer #CONVERTIR FRASES A PALABRAS
from keras.models import load_model 
#import JarvisAI
import pyttsx3
import speech_recognition as sr
from datetime import datetime
import pywhatkit as kit


rec = sr.Recognizer()
#asistente = JarvisAI.JarvisAssistant()

motor = pyttsx3.init('sapi5')

#velocidad a la que habla
motor.setProperty('rate',160)
#volumen
motor.setProperty('volume',1.5)
#voces
voces = motor.getProperty('voices')
print(voces)
motor.setProperty('voice', voces[0].id)

def hablar(texto):
    motor.say(texto)
    motor.runAndWait()

#SACAR LAS PALABRAS DE LAS ORACIONES
lematizador_español = SnowballStemmer("spanish")

intents = json.loads(open('intents.json').read())

#CREAR UNA VARIABLE PARA LAS PALABRAS CON UNA LIBRERIA
palabras = pickle.load(open('palabras.pkl','rb'))
clases = pickle.load(open('clases.pkl','rb'))

#CARGAR EL MODELO
modelo = load_model('chatbot.h5')

#PARA LIMPIAR LA ORACION, ANTES DE QUE RESPONDA EL CHAT BOT
def limpiar_oracion(oracion):
    palabras_oracion = nltk.word_tokenize(oracion)
    palabras_oracion = [lematizador_español.stem(palabra) for palabra in palabras_oracion]
    return palabras_oracion

#LE SALUDAS, Y VA A TRATAR DE RESPONDER CON LA ESRTUCTURA QUE HE DEFINIDO
#QUE SI EXISTE LA PALABRA DEL ENTRENAMIENTO, LO CONVIERTA EN 1 EN EL BAG
def verificar_palabras(oracion):
    palabras_oracion = limpiar_oracion(oracion)
    bag = [0]*len(palabras)
    for p in palabras_oracion:
        for i, palabra in enumerate(palabras):
            if palabra == p:
                bag[i] =1
    return np.array(bag)

#SE VA A PREDECIR LA CLASES DE LO QUE HA ESCRITO EL USUARIO
def predecir_clase(oracion):
    bag_palabras = verificar_palabras(oracion)
    respuesta = modelo.predict(np.array([bag_palabras]))[0]
    max_index = np.where(respuesta==np.max(respuesta))[0][0] #PARA SABER CUAL ES MAS PROBABLE DE LA ORACION, UNA CONDICIONAL
    clase = clases[max_index]
    return clase

#UNA FUNCION PARA OBTENER RESPUESTA POR PARTE DEL CHATBOT
def obtener_respuesta(clase, intents_json):
    lista_intents = intents_json['intents']
    resultado = ""
    for i in lista_intents:
        if i["tag"]==clase:
            resultado = random.choice(i['responses'])
            break
    return resultado

while True:
    with sr.Microphone() as origen:
        print("Escuchando...")
        audio = rec.listen(origen,10,3)

    print("Reconociendo...")
    #oracion = asistente.mic_input()
    oracion = rec.recognize_google(audio, language='es-ES')
    cl = predecir_clase(oracion)
    if cl=="hora": 
        hora = datetime.now().hour
        minutos = datetime.now().minute
        res = obtener_respuesta(cl, intents)
        res = res + " mira para que sepas la hora es " + str(hora) + " con " + str(minutos) + " minutos mongolon"
        print(res)
        hablar(res)
    elif cl=="youtube":
        res = obtener_respuesta(cl, intents)
        hablar(res)
        with sr.Microphone() as origen:
            print("Escuchando...")
            video = rec.listen(origen,10,3)
        videostr = rec.recognize_google(video, language='es-ES')
        hablar("Reproduciendo")
        kit.playonyt(videostr)
        print(videostr)
    else:
        res = obtener_respuesta(cl, intents)
        print(res)
        hablar(res)
    #asistente.text2speech(res)
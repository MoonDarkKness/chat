
import speech_recognition as sr

rec = sr.Recognizer()

with sr.Microphone() as origen:
    print("Escuchando...")
    audio = rec.listen(origen)

print("Reconociendo...")
texto = sr.recognize_google(audio, language='es-ES')
print(texto)



print("Reconociendo...")
texto = rec.recognize_google(audio, language='es-ES')
print(texto)
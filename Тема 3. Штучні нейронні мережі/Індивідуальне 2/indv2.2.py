import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#  Дані
# Вхідні ознаки: [Високий бал Мат, Високий бал Гум, Тех схильність, Творча ]
x_train = np.array([
    [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0],
    [1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0],
    [1, 1, 0, 1], [0, 1, 1, 1]
])
# Вихід: Спеціальність
y_train = np.array([
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
    [0, 1, 0, 0], [0, 1, 0, 0]
])

# Тестові дані
x_test = np.array([
    [1, 1, 0, 0], # Очікується КН або Біологія
    [0, 0, 1, 1], # Очікується Мистецтво або Право
    [0, 1, 1, 0], # Очікується Право або КН
    [1, 0, 0, 1]  # Очікується Біологія або Мистецво
])
# Очікувані мітки
y_test_labels = ['КН/Біологія', 'Мистецтво/Право', 'Право/КН', 'Біологія/Мистецтво']


specialties = ['Комп\'ютерні науки', 'Право', 'Мистецтво', 'Біологія']

# 2. Архітектура моделі
model = keras.Sequential(
    [
        keras.Input(shape=(4,), name="input_layer"),
        layers.Dense(6, activation="relu", name="hidden_layer"),
        layers.Dense(4, activation="softmax", name="output_layer"), # 4 вихідних нейрони
    ]
)
print("Архітектура моделі:")
model.summary()

# 3. Компіляція моделі

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 4. Навчання моделі
print("\nНавчання моделі...")

history = model.fit(x_train, y_train, batch_size=2, epochs=150, verbose=0)
print("Навчання завершено.")
print(f"Точність на навчальних даних: {history.history['accuracy'][-1]:.4f}")

# 5. Тестування моделі
print("\nТестування моделі...")
predictions = model.predict(x_test)

print("\nПередбачення (ймовірності для класів [КН, Право, Мистецтво, Біологія]):")
print(np.round(predictions, 3)) # Округл

print("\nРекомендовані спеціальності для тестових даних:")
for i, pred in enumerate(predictions):
    recommended_specialty_index = np.argmax(pred) #  індекс класу з найвищою ймовірністю
    recommended_specialty = specialties[recommended_specialty_index]
    print(f"Вхід: {x_test[i]} => Рекомендовано: {recommended_specialty} (Ймовірність: {pred[recommended_specialty_index]:.2f}). Очікувалось: {y_test_labels[i]}")
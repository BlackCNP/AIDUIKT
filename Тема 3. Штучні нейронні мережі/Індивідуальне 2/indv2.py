import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Таблиця істинності та дані
# Вхідні дані
X = np.array([
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
    [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
    [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
])

# Вихідні дані (значення функції Y = (X1 v X2) ^ X3 ^ X4)
Y = np.array([(x1 or x2) and x3 and x4 for x1, x2, x3, x4 in X])

print("--- Вхідні дані (X) ---")
print(X)
print("\n--- Очікувані вихідні дані (Y) ---")
print(Y)

# 2. Створення моделі MLP (4-2-1)
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='adam',
                    max_iter=10000, random_state=1, learning_rate_init=0.01)

# 3. Навчання нейронної мережі
print("\n--- Навчання мережі ---")
mlp.fit(X, Y)
print("Навчання завершено.")

# Виведення вагових коефіцієнтів
print("\n--- Вагові коефіцієнти ---")
print("Ваги між вхідним та прихованим шаром (4x2):")
print(mlp.coefs_[0])
print("\nВаги між прихованим та вихідним шаром (2x1):")
print(mlp.coefs_[1])
print("\nЗміщення (bias) для нейронів прихованого шару (2):")
print(mlp.intercepts_[0])
print("\nЗміщення (bias) для нейрона вихідного шару (1):")
print(mlp.intercepts_[1])

# 4. Тестування отриманої нейронної мережі
print("\n--- Тестування мережі ---")
predictions = mlp.predict(X)
print("Передбачені значення:")
print(predictions)
print("Очікувані значення:")
print(Y)

# 5. Аналіз роботи та якості навчання
accuracy = accuracy_score(Y, predictions)
print(f"\n Аналіз результатів ---")
print(f"Точність моделі: {accuracy * 100:.2f}%")

if accuracy == 1.0:
    print("Нейронна мережа успішно навчилася реалізовувати задану логічну функцію.")
else:
    print("Нейронна мережа допустила помилки при класифікації.")

    errors = np.where(Y != predictions)[0]
    print(f"Помилки на наступних вхідних векторах (індекси): {errors}")
    for i in errors:
      print(f"Вхід: {X[i]}, Очікуваний вихід: {Y[i]}, Передбачений вихід: {predictions[i]}")
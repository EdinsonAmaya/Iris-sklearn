from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def dibujar_iris_dispersion(iris):
    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
    ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
    )
    ax.set_title("Iris Dataset")
    plt.show()

def atributos_iris(iris):
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['especie'] = df['target'].apply(lambda x: iris.target_names[x])
    print(df.head())
    print(df.groupby('especie').size())
    df.describe()
    df.hist()
    plt.show()
    df_correlacion = df.drop(columns=['especie', 'target'])
    plt.figure(figsize=(8,6))
    sns.heatmap(df_correlacion.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlacion entre atributos')
    plt.show()
    df.boxplot(by='especie')
    plt.show()

def preparacion_datos(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    escaler = StandardScaler()
    X_train_scaled = escaler.fit_transform(X_train)
    X_test_scaled = escaler.transform(X_test)
    
    print("Datos preparados")

    return X_train_scaled, X_test_scaled, y_train, y_test

def entrenamiento_evaluacion(X_train_scaled, X_test_scaled, y_train, y_test, K_value = 10):
    
    knn_model = KNeighborsClassifier(n_neighbors=K_value)

    knn_model.fit(X_train_scaled, y_train)
    
    print("Entrenamiento completado")

    y_pred = knn_model.predict(X_test_scaled)
    prediccion = accuracy_score(y_test, y_pred)

    print("Precision del modelo: ", prediccion)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def ejecucion(iris):
    X = iris.data
    y = iris.target
    X_train_scaled, X_test_scaled, y_train, y_test = preparacion_datos(X, y)
    entrenamiento_evaluacion(X_train_scaled, X_test_scaled, y_train, y_test)


iris = datasets.load_iris()
#atributos_iris(iris)
ejecucion(iris)
#dibujar_iris_dispersion(iris)
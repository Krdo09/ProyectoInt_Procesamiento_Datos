from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]


# Parte I
def prom_age(dat: dataset):
    arr_age = np.array(dat['age'])
    print('The average age in dataset is: {:.2f}'
          .format(arr_age.mean()))


# Parte II
def to_df(dat: dataset) -> pd.DataFrame:
    df = pd.DataFrame(dat)
    return df


def separation(df: pd.DataFrame):
    is_dead = df[df['is_dead'] == 1]
    is_not_dead = df[df['is_dead'] == 0]
    return is_dead, is_not_dead


def average_age(df: pd.DataFrame) -> pd.DataFrame:
    mean_age = df['age'].mean()
    return mean_age


# Parte III
def data_check(df: pd.DataFrame):
    columns_names = ['age', 'anaemia', 'creatinine_phosphokinase',
                     'diabetes', 'ejection_fraction', 'high_blood_pressure',
                     'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                     'smoking', 'time', 'DEATH_EVENT']

    for column in columns_names:
        data_type = str(df[column].dtype)
        if data_type == 'int68':
            df[column] = pd.to_numeric(df[column], downcast='integer')

        if data_type == 'float64':
            df[column] = pd.to_numeric(df[column], downcast='float')

        if data_type == 'bool':
            df[column] = df[column].astype(bool)


def smoker(df: pd.DataFrame):
    smokers = df.groupby(['is_male', 'is_smoker']).size().reset_index(name='count')
    smokers = smokers[smokers['is_smoker']]
    smokers.columns = ['is_male', 'is_smoker', 'number_smokers']
    return smokers


# Parte IV
def api_request(source_url: str):
    response = requests.get(source_url)
    if response.status_code == requests.codes.ok:  # Verificamos que la respuesta de la pagina sea efectiva - 200
        # Decodificamos el contenido de la respuesta, ya que son datos numericos
        content = response.content.decode('utf-8')

        with open('data.csv', 'w', newline='\n') as csv:  # Escritura de los datos
            for line in content:
                csv.write(line)
        print('The data has loading')
    else:
        print(f'Something bad has happen, code error: {response.status_code}')


# Parte V
def data_empty_value_colum(df: pd.DataFrame):
    columns_names = df.columns

    # Rellenamos los valores faltantes con NaN y enviamos mensaje sí hay o no hay valores faltantes
    for colum in columns_names:
        df[f'{colum}'].fillna(np.nan, inplace=True)
        df_nan = df[df[f'{colum}'] == np.nan].value_counts().reset_index()
        # Comprobamos si hay valores faltantes
        if df_nan.empty:
            pass

        else:
            print(f'There are empty values in {colum}')
            return False

    return True


def repeated_data_cleaning(df: pd.DataFrame):
    # Verificamos si existen valores duplicados en el df
    duplicated = df.duplicated()
    duplicated = duplicated[duplicated].value_counts().reset_index()
    # Imprimimossi no hay filas repetidas o eliminamos si es lo contrario
    if duplicated.empty:
        print('There are not duplicated rows')
    else:
        df.drop_duplicates(inplace=True)
        print('The duplicated rows has remove')


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # La funcion revisara y eliminara las filas que las contengan
    columns_names = ['creatinine_phosphokinase', 'ejection_fraction',
                     'platelets', 'serum_creatinine', 'serum_sodium']

    # Calculamos los cuartiles Q1 y Q3 para cada columna
    for colum in columns_names:
        Q1 = df[f'{colum}'].quantile(0.25)
        Q3 = df[f'{colum}'].quantile(0.75)
        IQR = Q3 - Q1
        # Definimos limites para outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Eliminamos las filas con outliers
        df = df[(df[f'{colum}'] >= lower_bound) & (df[f'{colum}'] <= upper_bound)]

    print('The outliers are remove of DataFrame')
    return df


def age_category(df: pd.DataFrame) -> pd.DataFrame:
    # Definimos los intervalos para el recorte y categorizamos
    intervals = pd.IntervalIndex.from_tuples([(0, 12), (13, 19), (20, 39), (40, 59), (60, 120)])
    print(intervals)
    new_categories = pd.cut(df['age'], intervals, include_lowest=True)

    # Generamos las columnas dummies
    new_df = pd.get_dummies(new_categories, dtype=int)
    new_df.columns = ['Children', 'Teenager', 'Young Adult', 'Adult', 'Old Adult']

    # Unimos el df original con las nuevas columnas
    df_dummies = pd.concat([new_df, df], axis=1)

    return df_dummies


def data_processing(df_to_clean: pd.DataFrame):
    if data_empty_value_colum(df_to_clean):
        repeated_data_cleaning(df_to_clean)
        outliers_clean = remove_outliers(df_to_clean)

        print('The cleaning process has been successful')
        final_data = age_category(outliers_clean)
        final_data.to_csv('data_clean.csv', index=False)

    else:
        print('The data has a colum with a empty data. Please check the colum')


# Parte VII
def hist_age(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    plt.hist(df['age'], bins=15, edgecolor='black')
    plt.xlabel('Edad')
    plt.ylabel('Cantidad')
    plt.title('Distribución De Edades')
    plt.show()


def hist_group_graphic(df: pd.DataFrame):
    # Obtenemos la cantidad de anemicos por genero
    anaemia_group = df[['anaemia', 'sex']]
    anaemia_group = anaemia_group[anaemia_group['anaemia'] == 1].groupby(['sex']).value_counts().to_list()

    # Obtenemos la cantidad de diabeticos por genero
    diabetes_group = df[['diabetes', 'sex']]
    diabetes_group = diabetes_group[diabetes_group['diabetes'] == 1].groupby(['sex']).value_counts().to_list()

    # Obtenemos el grupo de fumadores
    smokers_group = df[['smoking', 'sex']]
    smokers_group = smokers_group[smokers_group['smoking'] == 1].groupby(['sex']).value_counts().to_list()

    # Obtenemos el grupo de personas que murieron durante el estudio
    deaths_group = df[['DEATH_EVENT', 'sex']]
    deaths_group = deaths_group[deaths_group['DEATH_EVENT'] == 1].groupby(['sex']).value_counts().to_list()

    # Separamos los datos en lista para mujeres y hombres
    group_names = ('Anémicos', 'Diabeticos', 'Fumadores', 'Muertos')
    woman_group = [anaemia_group[0], diabetes_group[0], smokers_group[0], deaths_group[0]]
    man_group = [anaemia_group[1], diabetes_group[1], smokers_group[1], deaths_group[1]]

    # Creamos la figura, el axis númerico y el tamaño de las barras
    fig, ax = plt.subplots(figsize=(10, 6))
    numerical_axis = np.arange(len(group_names))
    bar_width = 0.38

    # Creamos el grafico
    ax.bar(numerical_axis, man_group,
           width=bar_width, label='Hombres', color='blue')
    ax.bar(numerical_axis + bar_width, woman_group,
           width=bar_width, label='Mujeres', color='red')
    # Remplazamos los indices númericos por categoricos
    ax.set_xticks(numerical_axis + bar_width / 2)
    ax.set_xticklabels(group_names)

    # Agregamos detalles
    ax.set_title('Histograma Agrupado Por Sexo')
    ax.set_xlabel('Categorias')
    ax.set_ylabel('Cantidad')
    ax.legend()
    plt.show()


# Parte VIII
def pies_graphic(df: pd.DataFrame):
    # Obtenemos la cantidad de anemicos y no anemicos
    anaemia_group = df['anaemia'].value_counts().to_list()

    # Obtenemos la cantidad de diabeticos y no diabeticos
    diabetes_group = df['diabetes'].value_counts().to_list()

    # Obtenemos la cantidad de fumadores y no fumadores
    smokers_group = df['smoking'].value_counts().to_list()

    # Obtenemos a cantidad de muertos y vivos
    deaths_group = df['DEATH_EVENT'].value_counts().to_list()

    # Organizamos los datos
    true_groups = [anaemia_group[1], diabetes_group[1], smokers_group[1], deaths_group[1]]
    false_groups = [anaemia_group[0], diabetes_group[0], smokers_group[0], deaths_group[0]]
    names = ('Anémicos', 'Diabéticos', 'Fumadores', 'Muertos')
    category = ('No', 'Si')

    # Grafícamos
    fig, ax = plt.subplots(1, 4, figsize=(12, 10))
    for index, true, false, name in zip(range(0, 4), true_groups, false_groups, names):
        ax[index].pie([false, true], labels=category,
                      autopct='%1.1f%%',
                      startangle=90,
                      colors=['red', 'blue'])
        # Agreamos titulos
        ax[index].set_title(f'{name}')
    # Mostramos la grafíca
    plt.tight_layout()
    plt.show()


# Parte IX
def eliminate_colum(df: pd.DataFrame, colum_names: list[str], colum_save: list[int]):
    """
    Elimina las columnas de un DataFrame de pandas indicadas
    con el argumento colums_names. Si es necesario almacenar
    una o más columnas de las que se ván a eliminar del df,
    con el argumento colum_save enviar una lista con los
    indices de los nombres almacenados en colum_names.

    :param df: pd.DataFrame
    :param colum_names: list[str]
    :param colum_save: list[int]
    :return: tuple(pd.DataFrame, pd.Series | pd.DataFrame)
    """

    # Almacenamos las columnas si el argumento colum_save es utilizado
    if len(colum_save) != 0:
        to_save = []
        for index in colum_save:
            to_save.append(colum_names[index])

        finally_colum = df[to_save]

    # Eliminamos las columnas indicadas
    new_data = df.drop(df[colum_names], axis=1)

    return new_data, finally_colum


def scatter_graphic(df: pd.DataFrame):
    # Eliminamos las columnas y transformamos los datos
    to_eliminate = ['Children',
                    'Teenager',
                    'Young Adult',
                    'Adult',
                    'Old Adult',
                    'DEATH_EVENT']
    numpy_arrays, y = eliminate_colum(df, colum_names=to_eliminate, colum_save=[5])
    numpy_arrays = numpy_arrays.values
    y = y.values

    # Hacemos la transformación dimensional
    X_embedded = TSNE(
        n_components=3,
        learning_rate='auto',
        init='random',
        perplexity=3
    ).fit_transform(numpy_arrays)

    # Creamos la gráfica
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X_embedded[:, 0], y=X_embedded[:, 1], z=X_embedded[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=y[:, 0],
            colorscale='Viridis',
            opacity=0.8
        )
    ))

    fig.update_layout(
        title='Gráfico de dispersión 3D',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    fig.write_html('3d_graphic.html', auto_open=False)


# Parte X
def lin_regression_model(df: pd.DataFrame) -> (LinearRegression, float):
    # Eliminamos las columnas y obtenemos el vector
    to_eliminate = ['Children',
                    'Teenager',
                    'Young Adult',
                    'Adult',
                    'Old Adult',
                    'age',
                    'DEATH_EVENT']
    x, y = eliminate_colum(df, colum_names=to_eliminate, colum_save=[5])

    # Creamos el modelo y lo entrenamos
    model = LinearRegression()
    model.fit(x, y)

    # Calculamos R^2
    r_squared = model.score(x, y)

    return model, r_squared


def predictions(df: pd.DataFrame, model: LinearRegression):
    # Obtenemos los datos para las prediccionas
    to_eliminate = ['Children',
                    'Teenager',
                    'Young Adult',
                    'Adult',
                    'Old Adult',
                    'age',
                    'DEATH_EVENT']
    df_to_predictions, true_age = eliminate_colum(df, colum_names=to_eliminate, colum_save=[5])

    # Realizamos las prediciones
    new_data = df_to_predictions.iloc[:5, :]
    true_age = true_age.iloc[:5].values
    age_prediction = model.predict(new_data)

    print('Predictions:', age_prediction, sep='\n')
    print('True age:', true_age, sep='\n')

    return age_prediction, true_age


def lineal_regression_metrics(r_squared: float, y_predicted: list, y_true: list):
    # Calculamos el error cuadrático medio
    mse = mean_squared_error(y_true, y_predicted)
    print('Error cuadrático medio (MSE):', mse)
    print('Coeficiente de determinación:', r_squared)


def model_creation(df_to_model: pd.DataFrame):
    model, r_squared = lin_regression_model(df_to_model)
    y_predicted, y_true = predictions(df_to_model, model)
    lineal_regression_metrics(r_squared, y_predicted, y_true)


# Parte XI
def distribution_graphic(df: pd.DataFrame):
    # Obtemos la distribucion de las clases
    classes = ['Children', 'Teenager', 'Young Adult', 'Adult', 'Old Adult']
    df = df[classes]
    classes_distribution = []
    for name in classes:
        value = df[df[f'{name}'] == 1].value_counts()
        if value.empty:
            classes_distribution.append(0)
        else:
            value = value.to_list()
            classes_distribution.append(value[0])

    # Graficamos los datos
    plt.figure(figsize=(10, 8))
    plt.bar(classes, height=classes_distribution, width=0.7)
    plt.title('Distribución de las clases')
    plt.xlabel('Categorias')
    plt.ylabel('Cantidad')
    plt.show()


def tree_model(df: pd.DataFrame):
    # Eliminamos las columnas innecesarias
    to_eliminate = ['Children',
                    'Teenager',
                    'Young Adult',
                    'Adult',
                    'Old Adult']
    X, y = eliminate_colum(df, colum_names=to_eliminate, colum_save=[i for i in range(len(to_eliminate))])

    # Realizamos la partición
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Instanciamos nuestro modelo
    tree = DecisionTreeClassifier()
    # Entrenamos el modelo
    tree.fit(X_train, y_train)

    # Realizamos las predicciones
    y_pred_test = tree.predict(X_test)
    print("Precisión del modelo en test:", accuracy_score(y_test, y_pred_test))


# Parte XII
def random_forest(df: pd.DataFrame):
    # Eliminamos las columnas innecesarias
    to_eliminate = ['Children',
                    'Teenager',
                    'Young Adult',
                    'Adult',
                    'Old Adult']
    X, y = eliminate_colum(df, colum_names=to_eliminate, colum_save=[i for i in range(len(to_eliminate))])

    # Realizamos la partición
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Creamos el modelo
    r_forest = RandomForestClassifier(random_state=42)
    r_forest.fit(X_train, y_train)  # Entrenamos el modelo

    # Calculamos su matriz de confusion
    y_pred = r_forest.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    print('La matriz de confusion es:', matrix)

    # Calculamos el F1-Score y accuracy
    
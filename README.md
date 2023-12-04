# TimeSeriesCourse
 
## НАЗНАЧЕНИЕ

Данный репозиторий размещен по адресу https://github.com/mzym/TimeSeriesCourse/ и содержит материалы учебного курса "Анализ и прогнозирование временных рядов методами искусственного интеллекта".

## ПРЕПОДАВАТЕЛИ

Лектор: [Михаил Леонидович Цымлер](https://mzym.susu.ru), д.ф.-м.н. (mzym@susu.ru)

Ассистенты: 
* Андрей Игоревич Гоглачев (goglachevai@susu.ru)
* Яна Александровна Краева (kraevaya@susu.ru)
* Алексей Артемьевич Юртин (iurtinaa@susu.ru)

## СТРУКТУРА 

* _datasets_	Наборы данных для выполнения практических заданий
* _slides_	Слайды к лекциям курса
* _practice_	Практические задания и указания по их выполнению в форме Python-ноутбуков

## Working with Jupiter notebooks
To work with Jupyter notebooks, the project includes a docker container with Jupyter Lab. To run it:
1. fill the .env file based on the .env.example file
2. execute in terminal:
    ```bash
    docker-compose up --build -d notebooks
    ```
3. go to the url in your browser. This will open up a development environment in which you can conduct experiments.
    ```
    http://localhost:{JUPYTERLAB_PORT_FROM_ENV}/lab?token={JUPYTERLAB_TOKEN_FROM_ENV}
    ```

# ML-based Marketing

Руководитель: [Лукьянченко Пётр Павлович](https://www.hse.ru/staff/lukianchenko)

Студенты: [Гончаров Фёдор Михайлович](http://t.me/fmgoncharov), [Писцов Георгий Константинович](http://t.me/GoshaNice)

## Ресурсы

Используемые библиотеки:
- [CausalML](https://github.com/uber/causalml)
- [EconlML](https://github.com/py-why/EconML)
- [Gradio](https://github.com/gradio-app/gradio)

Отчеты:
- [КТ1](https://drive.google.com/file/d/1X9sN0MSib5yEGM-PElc6Y33UsGcLC2Nz/view?usp=share_link)
- [Итоговый отчёт](https://www.overleaf.com/read/zgnqqgmgcntm)

Ноутбуки:
- [Исследование и анализ данных](notebooks/ExplatoryDataAnalysis.ipynb)
- [Выбор модели и валидация](notebooks/Validation.ipynb)
- [Итоговая модель и интерфейс](notebooks/GradioInterface.ipynb)

Докер-контейнер

## Введение

В данной работе наша команда поставила перед собой задачу написать программу, способную выделять из всех клиентов компании восприимчивую к маркетинговым продуктам аудиторию, чтобы в последствии использовать эти данные при разработке и проведении маркетинговой стратегии

## Демонстрация

[![Watch the video](https://i.ibb.co/HDvkc70/tg-image-1095861058.jpg)](https://youtu.be/Rg-AUrIIauI)

## Результаты выбора модели и валидации

В качестве итогового продукта мы получили откалиброванную модель, работающую без сбоев, дающую хорошие показатели метрик качества.

Эксперименты с различными uplift-моделями показали, что для финального продукта разумнее всего в первую очередь использовать X-Learner с основой из `xgboost`. Реализации от двух рассматриваемых библиотек похожи по качеству, но EconML по большинству экспериментов работает быстрее, поэтому для финального реализации следует выбрать ее.

| Model      | CausalML score | CausalML time | EconML score        | EconML time |
|------------|----------------|---------------|---------------------|-------------|
| S XGBoost  | 3.562939e+12   | 175s          | 3.544060e+12        | 180s        |
| T XGBoost  | 3.816367e+12   | 167s          | 3.816367e+12        | 169s        |
| X XGBoost  | 3.809030e+12   | 497s          | **3.852631e+12**    | 331s        |
| S MLP      | 1.242707e+12   | 294s          | 1.275415e+12        | 310s        |
| T MLP      | 1.177970e+12   | 274s          | 1.177970e+12        | 297s        |
| X MLP      | 1.232555e+12   | 695s          | 1.233368e+12        | 589s        |
| S CatBoost | 2.782283e+12   | 21s           | 2.819787e+12        | 21s         |
| T CatBoost | 3.199669e+12   | 33s           | 3.199669e+12        | 33s         |
| X CatBoost | 3.226224e+12   | 209s          | 3.245900e+12        | 66s         |
| CTS        | 2.403556e+12   | 117s          | -                   | -           |
| CTS Forest | 2.111332e+12   | 640s          | 2.206507e+12        | 82s         |

## Распределение задач

| **Гончаров Фёдор**                                                  | **Писцов Георгий**                           |
|---------------------------------------------------------------------|----------------------------------------------|
| Разработка методологии экспериментов                                | Анализ и чистка исторических данных          |
| Эксперименты с Meta-Learners                                        | Эксперименты с CTS                           |
| Подготовка ТЗ для компании-партнёра                                 | Подготовка отчета по экспериментам           |
| Разработка фронтенда продукта                                       | Тестирование продукта                        |
| Подготовка графиков проектной работы                                | Подготовка текста проектной работы           |
| Значимость задачи                                                   | Актуальность задачи                          |
| Обзор Meta-Learners                                                 | Обзор SAM, CTS                               |
| Разбор EconML                                                       | Разбор CausalML                              |
| Разбор TripAdviser Case                                             | Разбор Bidder Case                           |
| Подготовка иллюстраций и таблиц                                     | Подготовка текста и вёрстки                  |

## Текущий продуктовый эксперимент

Наши исследования вызвали большой интерес со стороны компании, и было принято решение продолжить сотрудничество с целью оптимизации маркетинговых инструментов компании с использованием uplift-моделирования. В настоящее время мы проводим эксперимент на реальной аудитории для получения дополнительных данных о реакции пользователей на различные маркетинговые воздействия, направленные на увеличение их лояльности.

В ходе обсуждения программы лояльности компании было выявлено, что ключевым мотивирующим фактором для клиентов является угроза потери текущего статуса лояльности. С целью проверки этой гипотезы мы разработали и реализовали эксперимент, который включает отправку уведомлений о грядущем снижении статуса лояльности.

## Дизайн и реализация эксперимента

В рамках эксперимента, пользователи, у которых через две недели заканчивается срок действия текущего статуса лояльности, получают одно из трех уведомлений с одинаковой вероятностью. Все уведомления содержат информацию о предстоящем снижении статуса и необходимости совершить покупку для его сохранения. Отличие уведомлений заключается в дополнительных предложениях:

- Обычное уведомление. Эта группа пользователей служит контрольной в наших экспериментах. Данный тип воздействия уже применяется компанией и не предполагает дополнительных затрат.
- Уведомление с предложением \textbf{процентной скидки}. Подобное воздействие может быть эффективным для определенного сегмента клиентов, но его не стоит рассылать всем, так как для некоторых клиентов будет достаточно простого напоминания.
- Уведомление с предложением \textbf{купона на фиксированную сумму}. Этот тип воздействия является наиболее мощным, но и наиболее затратным для компании, поэтому его не стоит использовать по умолчанию для всех клиентов.
\end{enumerate}

![Схема эксперимента](https://i.ibb.co/0Qgtm2X/exp-teremok.png)

Результаты этого эксперимента позволят нам построить модель, с помощью которой для каждого клиента будет определяться наиболее подходящий тип маркетингового воздействия. Это поможет оптимизировать расходы на маркетинговые кампании и увеличить лояльность клиентов, что, в свою очередь, способствует повышению прибыльности компании.

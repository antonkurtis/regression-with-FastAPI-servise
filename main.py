import pandas as pd
import numpy as np
import random
import re
import sklearn
import pickle
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, Request, Form, File, UploadFile, Body
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error as MSE


random.seed(42)
np.random.seed(42)

app = FastAPI()

model = pickle.load(open("pickle/model.sav", 'rb'))
ohe = pickle.load(open("pickle/ohe.sav", 'rb'))
scaler = pickle.load(open("pickle/scaler.sav", 'rb'))


app.mount("/files", StaticFiles(directory="files"), name="files")

templates = Jinja2Templates(directory="templates")


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: np.float


class Items(BaseModel):
    objects: List[Item]


@app.get("/")
async def root(request:Request, message="HELLO, CAR'S DEALER!"):
    return templates.TemplateResponse("index.html",
                                      {"request": request,
                                       "message": message})


@app.get("/get_item")
async def get_item(request:Request):
    docs = os.listdir('files/')
    return templates.TemplateResponse('predict_item.html',
                                      {"request": request,
                                       "doc": docs[0]})


@app.get("/get_items")
async def get_item(request:Request):
    docs = os.listdir('files/')
    return templates.TemplateResponse('predict_item.html',
                                      {"request": request,
                                       "doc": docs[3]})


@app.post("/predict_item")
async def predict_item(request: Request,
                   name = Form(...),
                   year = Form(...),
                   selling_price = Form(...),
                   km_driven = Form(...),
                   fuel = Form(...),
                   seller_type = Form(...),
                   transmission = Form(...),
                   owner = Form(...),
                   mileage = Form(...),
                   engine = Form(...),
                   max_power = Form(...),
                   torque = Form(...),
                   seats = Form(...)):

    car = {'name': str(name),'year': int(year),'selling_price': int(selling_price),'km_driven': int(km_driven),
           'fuel': str(fuel),'seller_type': str(seller_type),
           "transmission": str(transmission),'owner': str(owner),'mileage': str(mileage),'engine': str(engine),
           'max_power': str(max_power),'torque': str(torque),'seats': np.float(seats)}
    car_df = pd.DataFrame(car, index=[0])
    car_df.to_csv('files/car.csv')

    return 'Params of Car uploaded. Please, go back to start page'


@app.post("/predict_items")
async def predict_items(request: Request,
                        name: str = Form(...),
                        car_file: UploadFile = File(...)):
    file_name = 'list_of_cars' + '.csv'
    save_path = f'files/{file_name}'
    with open(save_path, 'wb') as f:
        for line in car_file.file:
            f.write(line)
    return 'List of Car uploaded. Please, go back to start page'


#Ф-ия для чистки колонок mileage, engine, max_power
def get_nums(column):
    if type(column) == float:
      return None
    elif len(re.findall(r'\d+',column.strip())) == 0:
      return None
    elif column.strip() == '0':
      return None
    else:
      return float('.'.join(re.findall(r'\d+',column.strip())))


#Функция, которая достает значение крутящего момента из torque
def get_torque(col):
  if type(col) == float:
    return None
  elif re.findall(r'\d+\s\W\s\d+', col):
    return col.split('/')[0].strip()
  elif re.findall(r'\d+\snm\s/\d+\srpm', col.lower()):
    return ''.join(col.lower().split('/')[0].strip().split())
  elif re.findall(r'\d+nm@|\d+\W\d+nm@', col.lower()):
    return col.lower().split()[0].strip()
  elif re.findall(r'\d+@\s\d+\W\d+rpm|\d+@\s\d+rpm', col.lower()):
    return col.lower().split()[0].strip()
  elif re.findall(r'\d+\s+nm\s|\d+\W\d+\s+nm\s', col.lower()):
    return ''.join(col.lower().split()[0:2]).strip()
  elif re.findall(r'\d+nm.\d+\W\d+kgm.', col.lower()):
    return col.lower().split('@')[0]
  elif re.findall(r'\d+kgm@\s|\d+\W\d+kgm@\s', col.lower()):
    return col.lower().split()[0]
  elif re.findall(r'\d+\skgm\s|\d+\W\d+\skgm\s', col.lower()):
    return ''.join(col.lower().split()[0:2]).strip()
  elif re.findall(r'.kgm@ rpm.', col.lower()):
    return col.lower().split('@')[0] + 'kgm'
  elif re.findall(r'\d+nm\sat|\d+\W\d+nm\sat', col.lower()):
    return col.lower().split()[0]
  elif re.findall(r'\d+nm|\d+\W\d+nm', col.lower()):
    return col.lower()
  elif re.findall(r'\d+@\s\d+\W\d+|\d+\W\d+@\s\d+\W\d+', col.lower()):
    return col.lower().split('@')[0]


#Функция, которая достает значение оборотов из torque
def max_torque_rpm(col):
  if type(col) == float:
    return None
  elif re.findall(r'\d+\snm\s/\d+\srpm', col.lower()):
    return ''.join(col.lower().split('/')[1].strip().split())
  elif re.findall(r'\d+\s\W\s\d+', col):
    return col.split('/')[1].strip()
  elif re.findall(r'\d+rpm', col.lower()):
    return col.lower().split()[-1].strip()
  elif re.findall(r'\d+\s+rpm', col.lower()):
    return ''.join(col.lower().split('at')[-1].split())
  elif re.findall(r'.kgm@ rpm.', col.lower()):
    return col.split()[1].split('(')[0] + 'rpm'


#избавимся от бесящей @. Правда не знаю зачем :)
def drop_fkng(col):
  if col is None:
    return np.nan
  else:
    return col.replace('@', '')


car = pd.read_csv('files/car.csv')

cars = pd.read_csv('files/list_of_cars.csv')
cars_list = pd.read_csv('files/list_of_cars.csv')


car['mileage'] = car.mileage.apply(get_nums)
car['engine'] = car.engine.apply(get_nums)
car['max_power'] = car.max_power.apply(get_nums)

cars['mileage'] = cars.mileage.apply(get_nums)
cars['engine'] = cars.engine.apply(get_nums)
cars['max_power'] = cars.max_power.apply(get_nums)


car['max_torque_rpm'] = car.torque.apply(max_torque_rpm)
car['torque'] = car.torque.apply(get_torque)
car['torque'] = car.torque.apply(drop_fkng)

cars['max_torque_rpm'] = cars.torque.apply(max_torque_rpm)
cars['torque'] = cars.torque.apply(get_torque)
cars['torque'] = cars.torque.apply(drop_fkng)


#В колонке torque разные единицы измерения. Переведем кг\м в Нм
#1 кг\м ~ 9.81 Нм
def kgm_to_nm(col):
  if type(col) == float:
    return np.nan
  elif re.findall(r'\d+nm.\d+\W\d+\w+.', col):
    return float(re.findall(r'\d+|\d+\W\d+', col.split('(')[0])[0])
  elif re.findall(r'\d+kgm', col):
    if len(re.findall(r'\d+|\d+\W\d+', col)) == 1:
      return float(re.findall(r'\d+|\d+\W\d+', col)[0]) * 9.81
    elif len(re.findall(r'\d+|\d+\W\d+', col)) == 2:
      return float('.'.join(re.findall(r'\d+|\d+\W\d+', col))) * 9.81
  elif re.findall(r'\d+nm', col):
    if len(re.findall(r'\d+|\d+\W\d+', col)) == 1:
      return float(re.findall(r'\d+|\d+\W\d+', col)[0])
    elif len(re.findall(r'\d+|\d+\W\d+', col)) == 2:
      return float('.'.join(re.findall(r'\d+|\d+\W\d+', col)))
  else:
    return float(col)


#Почистим колонку max_torque_rpm
#там, где значение оборотов указано диапазоном, возьмем среднее в нем
def clear_max_torque_rpm(col):
  if col is None:
    return np.nan
  else:
    val = col.replace(',', '').replace('+/-500', '').replace('rpm', '').replace('~', '-')
    if len(val.split('-')) == 1:
      return float(val)
    elif len(val.split('-')) == 2:
      return np.mean(list(map(int, val.split('-'))))


car['torque'] = car.torque.apply(kgm_to_nm)
car['max_torque_rpm'] = car.max_torque_rpm.apply(clear_max_torque_rpm)

cars['torque'] = cars.torque.apply(kgm_to_nm)
cars['max_torque_rpm'] = cars.max_torque_rpm.apply(clear_max_torque_rpm)


#Спарсим марку авто
def auto_brend(col):
  return col.split()[0]


car['brand'] = car.name.apply(auto_brend)
cars['brand'] = cars.name.apply(auto_brend)


cars.seats.fillna(cars.seats.mode()[0], inplace=True)
df_train = cars.fillna(cars.mean())

car['seats'] = car.seats.astype(int).astype(str)
cars['seats'] = cars.seats.astype(int).astype(str)


#Добавим предложенное отношение числа "лошадей" на литр объема
car['power_per_eng'] = (car.max_power / car.engine) * 1000
cars['power_per_eng'] = (cars.max_power / cars.engine) * 1000

#Добавим предложенный квадрат года
car['year^2'] = car.year ** 2
cars['year^2'] = cars.year ** 2


car = car.drop(['selling_price', 'name'], axis = 1)
cars = cars.drop(['name'], axis = 1)

#отлогарифмируем некоторые из признаков и добавим их
car['torque_log'] = car.torque.apply(np.log)
car['max_power_log'] = car.max_power.apply(np.log)
car['km_driven_log'] = car.km_driven.apply(np.log)
car['year_log'] = car.year.apply(np.log)
car['year^2_log'] = car['year^2'].apply(np.log)
car['power_per_eng_log'] = car.power_per_eng.apply(np.log)

cars['torque_log'] = cars.torque.apply(np.log)
cars['max_power_log'] = cars.max_power.apply(np.log)
cars['km_driven_log'] = cars.km_driven.apply(np.log)
cars['year_log'] = cars.year.apply(np.log)
cars['year^2_log'] = cars['year^2'].apply(np.log)
cars['power_per_eng_log'] = cars.power_per_eng.apply(np.log)

cat_car = car.select_dtypes(include='object')
num_car = car.select_dtypes(exclude='object')

cat_car = pd.DataFrame(ohe.transform(cat_car))
num_car = pd.DataFrame(scaler.transform(num_car), columns=num_car.columns)

x_car = pd.concat([num_car, cat_car], axis = 1)


cat_cars = cars.select_dtypes(include='object')
num_cars = cars.select_dtypes(exclude='object')


cat_cars = pd.DataFrame(ohe.transform(cat_cars))
num_cars = pd.DataFrame(scaler.transform(num_cars), columns=num_cars.columns)

x_cars= pd.concat([num_cars, cat_cars], axis = 1)

x_cars = x_cars.fillna(x_cars.mean())
x_car = x_car.fillna(0)


pred_one_car = model.predict(x_car)

f = open('files/Predict_one_car.txt', 'w')
f.write(str(pred_one_car[0]))
f.close()


pred_cars = pd.DataFrame(model.predict(x_cars), columns=['selling_price'])

predict_list_of_cars = pd.concat([cars_list, pred_cars], axis=1)
predict_list_of_cars.to_csv('files/Predict_list_of_cars.csv')

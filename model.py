{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Year</th>\n",
       "      <th>Kilometers_Driven</th>\n",
       "      <th>Seats</th>\n",
       "      <th>Price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>6019.000000</td>\n",
       "      <td>6.019000e+03</td>\n",
       "      <td>5977.000000</td>\n",
       "      <td>6019.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2013.358199</td>\n",
       "      <td>5.873838e+04</td>\n",
       "      <td>5.278735</td>\n",
       "      <td>9.479468</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>3.269742</td>\n",
       "      <td>9.126884e+04</td>\n",
       "      <td>0.808840</td>\n",
       "      <td>11.187917</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1998.000000</td>\n",
       "      <td>1.710000e+02</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.440000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2011.000000</td>\n",
       "      <td>3.400000e+04</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>3.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>2014.000000</td>\n",
       "      <td>5.300000e+04</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.640000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>2016.000000</td>\n",
       "      <td>7.300000e+04</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>9.950000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2019.000000</td>\n",
       "      <td>6.500000e+06</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>160.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Year  Kilometers_Driven        Seats        Price\n",
       "count  6019.000000       6.019000e+03  5977.000000  6019.000000\n",
       "mean   2013.358199       5.873838e+04     5.278735     9.479468\n",
       "std       3.269742       9.126884e+04     0.808840    11.187917\n",
       "min    1998.000000       1.710000e+02     0.000000     0.440000\n",
       "25%    2011.000000       3.400000e+04     5.000000     3.500000\n",
       "50%    2014.000000       5.300000e+04     5.000000     5.640000\n",
       "75%    2016.000000       7.300000e+04     5.000000     9.950000\n",
       "max    2019.000000       6.500000e+06    10.000000   160.000000"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_csv(\"245550_518431_bundle_archive//train-data.csv\",index_col=0)\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['New_Price']=df.New_Price.str.extract('(\\d+)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Name                    0\n",
       "Location                0\n",
       "Year                    0\n",
       "Kilometers_Driven       0\n",
       "Fuel_Type               0\n",
       "Transmission            0\n",
       "Owner_Type              0\n",
       "Mileage                 2\n",
       "Engine                 36\n",
       "Power                  36\n",
       "Seats                  42\n",
       "New_Price            5195\n",
       "Price                   0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1876"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df['Name'].unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Hyderabad', 'Jaipur',\n",
       "       'Kochi', 'Kolkata', 'Delhi', 'Bangalore', 'Ahmedabad'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Location'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2010, 2015, 2011, 2012, 2013, 2016, 2018, 2014, 2017, 2007, 2009,\n",
       "       2008, 2019, 2006, 2005, 2004, 2002, 2000, 2003, 1999, 2001, 1998],\n",
       "      dtype=int64)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Year.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['998 CC', '1582 CC', '1199 CC', '1248 CC', '1968 CC', '814 CC',\n",
       "       '1461 CC', '2755 CC', '1598 CC', '1462 CC', '1497 CC', '2179 CC',\n",
       "       '2477 CC', '1498 CC', '2143 CC', '1995 CC', '1984 CC', '1197 CC',\n",
       "       '2494 CC', '1798 CC', '2696 CC', '2698 CC', '1061 CC', '1198 CC',\n",
       "       '2987 CC', '796 CC', '624 CC', '1999 CC', '1991 CC', '2694 CC',\n",
       "       '1120 CC', '2498 CC', '799 CC', '2393 CC', '1399 CC', '1796 CC',\n",
       "       '2148 CC', '1396 CC', '1950 CC', '4806 CC', '1998 CC', '1086 CC',\n",
       "       '1193 CC', '2982 CC', '1493 CC', '2967 CC', '2993 CC', '1196 CC',\n",
       "       '1799 CC', '2497 CC', '2354 CC', '1373 CC', '2996 CC', '1591 CC',\n",
       "       '2894 CC', '5461 CC', '1595 CC', '936 CC', '1997 CC', nan,\n",
       "       '1896 CC', '1390 CC', '1364 CC', '2199 CC', '993 CC', '999 CC',\n",
       "       '1405 CC', '2956 CC', '1794 CC', '995 CC', '2496 CC', '1599 CC',\n",
       "       '2400 CC', '1495 CC', '2523 CC', '793 CC', '4134 CC', '1596 CC',\n",
       "       '1395 CC', '2953 CC', '1586 CC', '2362 CC', '1496 CC', '1368 CC',\n",
       "       '1298 CC', '1956 CC', '1299 CC', '3498 CC', '2835 CC', '1150 CC',\n",
       "       '3198 CC', '1343 CC', '1499 CC', '1186 CC', '1590 CC', '2609 CC',\n",
       "       '2499 CC', '2446 CC', '1978 CC', '2360 CC', '3436 CC', '2198 CC',\n",
       "       '4367 CC', '2706 CC', '1422 CC', '2979 CC', '1969 CC', '1489 CC',\n",
       "       '2489 CC', '1242 CC', '1388 CC', '1172 CC', '2495 CC', '1194 CC',\n",
       "       '3200 CC', '1781 CC', '1341 CC', '2773 CC', '3597 CC', '1985 CC',\n",
       "       '2147 CC', '1047 CC', '2999 CC', '2995 CC', '2997 CC', '1948 CC',\n",
       "       '2359 CC', '4395 CC', '2349 CC', '2720 CC', '1468 CC', '3197 CC',\n",
       "       '2487 CC', '1597 CC', '2771 CC', '72 CC', '4951 CC', '970 CC',\n",
       "       '2925 CC', '2200 CC', '5000 CC', '2149 CC', '5998 CC', '2092 CC',\n",
       "       '5204 CC', '2112 CC', '1797 CC'], dtype=object)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Engine.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['26.6 km/kg', '19.67 kmpl', '18.2 kmpl', '20.77 kmpl', '15.2 kmpl',\n",
       "       '21.1 km/kg', '23.08 kmpl', '11.36 kmpl', '20.54 kmpl',\n",
       "       '22.3 kmpl', '21.56 kmpl', '16.8 kmpl', '25.2 kmpl', '12.7 kmpl',\n",
       "       '0.0 kmpl', '13.5 kmpl', '25.8 kmpl', '28.4 kmpl', '20.45 kmpl',\n",
       "       '14.84 kmpl', '22.69 kmpl', '23.65 kmpl', '13.53 kmpl',\n",
       "       '18.5 kmpl', '14.4 kmpl', '20.92 kmpl', '17.5 kmpl', '12.8 kmpl',\n",
       "       '19.01 kmpl', '14.53 kmpl', '11.18 kmpl', '12.4 kmpl',\n",
       "       '16.09 kmpl', '14.0 kmpl', '24.3 kmpl', '18.15 kmpl', '11.74 kmpl',\n",
       "       '22.07 kmpl', '19.7 kmpl', '25.4 kmpl', '25.32 kmpl', '14.62 kmpl',\n",
       "       '14.28 kmpl', '14.9 kmpl', '11.25 kmpl', '24.4 kmpl', '16.55 kmpl',\n",
       "       '17.11 kmpl', '22.9 kmpl', '17.8 kmpl', '18.9 kmpl', '15.04 kmpl',\n",
       "       '25.17 kmpl', '20.36 kmpl', '13.29 kmpl', '13.68 kmpl',\n",
       "       '20.0 kmpl', '15.8 kmpl', '25.0 kmpl', '16.4 kmpl', '24.52 kmpl',\n",
       "       '22.1 kmpl', '8.5 kmpl', '15.1 kmpl', '16.95 kmpl', '19.64 kmpl',\n",
       "       '16.5 kmpl', '18.53 kmpl', '17.57 kmpl', '18.0 kmpl', '23.2 kmpl',\n",
       "       '16.73 kmpl', '17.0 kmpl', '13.0 kmpl', '17.68 kmpl', '22.7 kmpl',\n",
       "       '16.2 kmpl', '15.26 kmpl', '23.0 kmpl', '19.83 kmpl', '14.94 kmpl',\n",
       "       '17.71 kmpl', '14.74 kmpl', '16.0 kmpl', '22.32 kmpl',\n",
       "       '12.99 kmpl', '23.3 kmpl', '19.15 kmpl', '10.8 kmpl', '15.0 kmpl',\n",
       "       '22.0 kmpl', '21.9 kmpl', '12.05 kmpl', '11.7 kmpl', '21.21 kmpl',\n",
       "       '20.73 kmpl', '21.1 kmpl', '24.07 kmpl', '19.0 kmpl', '20.58 kmpl',\n",
       "       '19.27 kmpl', '11.5 kmpl', '18.6 kmpl', '21.14 kmpl', '11.05 kmpl',\n",
       "       '21.76 kmpl', '7.81 kmpl', '21.66 kmpl', '17.2 kmpl', '20.63 kmpl',\n",
       "       '19.4 kmpl', '14.8 kmpl', '26.0 kmpl', '20.4 kmpl', '21.5 kmpl',\n",
       "       '15.3 kmpl', '17.9 kmpl', '16.6 kmpl', '22.54 kmpl', '25.44 kmpl',\n",
       "       '13.7 kmpl', '22.48 kmpl', '12.9 kmpl', '19.98 kmpl', '21.4 kmpl',\n",
       "       '19.81 kmpl', '15.4 kmpl', '25.47 kmpl', '19.87 kmpl',\n",
       "       '17.45 kmpl', '14.7 kmpl', '15.64 kmpl', '15.73 kmpl',\n",
       "       '23.59 kmpl', '16.1 kmpl', '27.4 kmpl', '20.46 kmpl', '15.29 kmpl',\n",
       "       '20.51 kmpl', '11.8 kmpl', '14.3 kmpl', '14.67 kmpl', '17.19 kmpl',\n",
       "       '21.03 kmpl', '22.5 kmpl', '16.82 kmpl', '11.72 kmpl', '17.4 kmpl',\n",
       "       '17.05 kmpl', '24.0 kmpl', '28.09 kmpl', '20.5 kmpl', '13.1 kmpl',\n",
       "       '19.91 kmpl', '18.7 kmpl', '16.38 kmpl', '11.57 kmpl', '17.3 kmpl',\n",
       "       '22.95 kmpl', '18.88 kmpl', '23.4 kmpl', '22.74 kmpl',\n",
       "       '12.07 kmpl', '17.1 kmpl', '18.48 kmpl', '16.47 kmpl', '23.1 kmpl',\n",
       "       '14.07 kmpl', '16.02 kmpl', '19.3 kmpl', '17.7 kmpl', '9.52 kmpl',\n",
       "       '14.75 kmpl', '26.3 km/kg', '11.3 kmpl', '21.12 kmpl',\n",
       "       '21.02 kmpl', '14.45 kmpl', '19.33 kmpl', '13.8 kmpl', '24.7 kmpl',\n",
       "       '11.0 kmpl', '11.07 kmpl', '21.43 kmpl', '14.21 kmpl',\n",
       "       '18.86 kmpl', '16.07 kmpl', '13.49 kmpl', '20.38 kmpl',\n",
       "       '12.0 kmpl', '17.01 kmpl', '13.2 kmpl', '20.37 kmpl', '15.1 km/kg',\n",
       "       '15.96 kmpl', '14.16 kmpl', '13.17 kmpl', '27.62 kmpl',\n",
       "       '25.1 kmpl', '15.17 kmpl', '11.33 kmpl', '17.92 kmpl',\n",
       "       '12.55 kmpl', '12.6 kmpl', '17.72 kmpl', '18.16 kmpl',\n",
       "       '15.68 kmpl', '15.5 kmpl', '12.1 kmpl', '14.83 kmpl', '17.6 kmpl',\n",
       "       '14.6 kmpl', '14.66 kmpl', '10.93 kmpl', '20.68 kmpl', '9.9 kmpl',\n",
       "       '21.13 kmpl', '20.14 kmpl', '19.2 kmpl', '27.3 kmpl', '16.36 kmpl',\n",
       "       '26.59 kmpl', '12.5 kmpl', '13.6 kmpl', '15.06 kmpl', '10.13 kmpl',\n",
       "       '17.21 kmpl', '15.97 kmpl', '10.5 kmpl', '14.69 kmpl', '23.9 kmpl',\n",
       "       '19.1 kmpl', '21.27 kmpl', '15.9 kmpl', '20.7 kmpl', '14.1 kmpl',\n",
       "       '20.89 kmpl', '18.12 kmpl', '12.3 kmpl', '19.71 kmpl', '9.43 kmpl',\n",
       "       '13.4 kmpl', '13.14 kmpl', '18.1 kmpl', '22.77 kmpl', '14.49 kmpl',\n",
       "       '12.39 kmpl', '10.91 kmpl', '20.85 kmpl', '15.63 kmpl',\n",
       "       '27.39 kmpl', '18.3 kmpl', '16.78 kmpl', '25.5 kmpl', '10.0 kmpl',\n",
       "       '13.73 kmpl', '24.2 kmpl', '14.02 kmpl', '26.83 km/kg',\n",
       "       '16.77 kmpl', '24.5 kmpl', '20.34 kmpl', '21.7 kmpl', '9.7 kmpl',\n",
       "       '14.33 kmpl', '21.64 kmpl', '13.2 km/kg', '19.16 kmpl',\n",
       "       '16.93 kmpl', '9.0 kmpl', '26.2 km/kg', '16.3 kmpl', '12.62 kmpl',\n",
       "       '17.3 km/kg', '20.64 kmpl', '14.24 kmpl', '18.06 kmpl',\n",
       "       '10.2 kmpl', '10.1 kmpl', '18.25 kmpl', '13.93 kmpl', '25.83 kmpl',\n",
       "       '8.6 kmpl', '13.24 kmpl', '17.09 kmpl', '23.84 kmpl', '8.45 kmpl',\n",
       "       '19.6 kmpl', '19.5 kmpl', '20.3 kmpl', '16.05 kmpl', '11.2 kmpl',\n",
       "       '27.03 kmpl', '18.78 kmpl', '12.35 kmpl', '14.59 kmpl',\n",
       "       '17.32 kmpl', '14.95 kmpl', '13.22 kmpl', '23.03 kmpl',\n",
       "       '33.44 km/kg', '15.6 kmpl', '19.12 kmpl', '10.98 kmpl',\n",
       "       '33.54 km/kg', '16.46 kmpl', '18.4 kmpl', '11.1 kmpl',\n",
       "       '13.01 kmpl', '18.8 kmpl', '16.52 kmpl', '18.44 kmpl',\n",
       "       '19.49 kmpl', '23.5 kmpl', '23.8 kmpl', '12.65 kmpl', '20.65 kmpl',\n",
       "       '21.72 kmpl', '12.19 kmpl', '26.1 kmpl', '18.33 kmpl',\n",
       "       '12.81 kmpl', '17.5 km/kg', '17.06 kmpl', '17.67 kmpl',\n",
       "       '19.34 kmpl', '8.3 kmpl', '16.96 kmpl', '11.79 kmpl', '20.86 kmpl',\n",
       "       '16.98 kmpl', '11.68 kmpl', '15.74 kmpl', '15.7 kmpl',\n",
       "       '18.49 kmpl', '10.9 kmpl', '19.59 kmpl', '11.4 kmpl', '13.06 kmpl',\n",
       "       '21.0 kmpl', '15.15 kmpl', '16.9 kmpl', '18.23 kmpl', '25.0 km/kg',\n",
       "       '17.16 kmpl', '17.43 kmpl', '19.08 kmpl', '18.56 kmpl',\n",
       "       '11.9 kmpl', '24.6 km/kg', '21.79 kmpl', '12.95 kmpl', '25.6 kmpl',\n",
       "       '13.45 km/kg', '26.21 kmpl', '13.58 kmpl', '16.25 kmpl',\n",
       "       '10.4 kmpl', '17.44 kmpl', '19.2 km/kg', '22.71 kmpl',\n",
       "       '17.54 kmpl', '22.1 km/kg', '17.0 km/kg', '15.87 kmpl', '9.5 kmpl',\n",
       "       '11.56 kmpl', '14.39 kmpl', '19.09 kmpl', '17.85 kmpl',\n",
       "       '31.79 km/kg', '18.18 kmpl', '21.19 kmpl', '21.8 kmpl',\n",
       "       '15.42 kmpl', '14.47 kmpl', '19.69 kmpl', '12.83 kmpl', '8.0 kmpl',\n",
       "       '22.8 km/kg', '12.63 kmpl', '14.57 kmpl', '27.28 kmpl',\n",
       "       '15.41 kmpl', '32.26 km/kg', '18.19 kmpl', '13.33 kmpl',\n",
       "       '16.7 kmpl', '17.84 kmpl', '20.0 km/kg', '23.19 kmpl',\n",
       "       '11.49 kmpl', '18.51 kmpl', '13.44 kmpl', '8.7 kmpl', '8.77 kmpl',\n",
       "       '17.97 kmpl', '23.57 kmpl', '12.37 kmpl', '9.1 kmpl', '12.51 kmpl',\n",
       "       '19.44 kmpl', '21.38 kmpl', '16.51 kmpl', '24.8 kmpl',\n",
       "       '14.42 kmpl', '14.53 km/kg', '26.8 kmpl', '24.04 kmpl', '9.8 kmpl',\n",
       "       '19.68 kmpl', '21.4 km/kg', '21.2 kmpl', '19.72 kmpl', '14.2 kmpl',\n",
       "       '12.98 kmpl', '23.01 kmpl', '16.12 kmpl', '9.3 kmpl', '15.85 kmpl',\n",
       "       nan, '17.88 kmpl', '10.6 kmpl', '11.78 kmpl', '7.94 kmpl',\n",
       "       '25.01 kmpl', '8.1 kmpl', '13.9 kmpl', '11.62 kmpl', '20.62 kmpl',\n",
       "       '15.11 kmpl', '10.37 kmpl', '18.59 kmpl', '9.74 kmpl',\n",
       "       '14.81 kmpl', '8.2 kmpl', '12.97 kmpl', '7.5 kmpl', '30.46 km/kg',\n",
       "       '6.4 kmpl', '12.85 kmpl', '18.69 kmpl', '17.24 kmpl'], dtype=object)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Mileage.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['58.16 bhp', '126.2 bhp', '88.7 bhp', '88.76 bhp', '140.8 bhp',\n",
       "       '55.2 bhp', '63.1 bhp', '171.5 bhp', '103.6 bhp', '74 bhp',\n",
       "       '103.25 bhp', '116.3 bhp', '187.7 bhp', '115 bhp', '175.56 bhp',\n",
       "       '98.6 bhp', '83.8 bhp', '167.62 bhp', '190 bhp', '88.5 bhp',\n",
       "       '177.01 bhp', '80 bhp', '67.1 bhp', '102 bhp', '108.45 bhp',\n",
       "       '138.1 bhp', '184 bhp', '179.5 bhp', '103.5 bhp', '64 bhp',\n",
       "       '82 bhp', '254.8 bhp', '73.9 bhp', '46.3 bhp', '37.5 bhp',\n",
       "       '77 bhp', '82.9 bhp', '149.92 bhp', '138.03 bhp', '112.2 bhp',\n",
       "       '163.7 bhp', '71 bhp', '105 bhp', '174.33 bhp', '75 bhp',\n",
       "       '103.2 bhp', '53.3 bhp', '78.9 bhp', '147.6 bhp', '147.8 bhp',\n",
       "       '68 bhp', '186 bhp', '170 bhp', '69 bhp', '140 bhp', '78 bhp',\n",
       "       '194 bhp', '500 bhp', '108.5 bhp', '86.8 bhp', '187.74 bhp',\n",
       "       'null bhp', '132 bhp', '86.7 bhp', '73.94 bhp', '117.3 bhp',\n",
       "       '218 bhp', '168.5 bhp', '89.84 bhp', '110 bhp', '90 bhp',\n",
       "       '82.85 bhp', '67 bhp', '241.4 bhp', '35 bhp', '270.9 bhp',\n",
       "       '126.32 bhp', '73 bhp', '130 bhp', '100.6 bhp', '150 bhp',\n",
       "       '75.94 bhp', '215 bhp', '107.3 bhp', '37.48 bhp', '120 bhp',\n",
       "       '178 bhp', '152 bhp', '91.1 bhp', '85.80 bhp', '362.07 bhp',\n",
       "       '121.3 bhp', '143 bhp', '81.80 bhp', '171 bhp', '76.8 bhp',\n",
       "       '103.52 bhp', '444 bhp', '362.9 bhp', '67.06 bhp', '120.7 bhp',\n",
       "       '258 bhp', '81.86 bhp', '112 bhp', '88.73 bhp', '57.6 bhp',\n",
       "       '157.75 bhp', '102.5 bhp', '201.1 bhp', '83.1 bhp', '68.05 bhp',\n",
       "       '88.50 bhp', nan, '106 bhp', '100 bhp', '81.83 bhp', '85 bhp',\n",
       "       '64.1 bhp', '177.5 bhp', '246.7 bhp', '177.46 bhp', '65 bhp',\n",
       "       '67.04 bhp', '189.08 bhp', '99 bhp', '53.5 bhp', '194.3 bhp',\n",
       "       '70 bhp', '183 bhp', '254.79 bhp', '66.1 bhp', '76 bhp', '60 bhp',\n",
       "       '123.24 bhp', '47.3 bhp', '118 bhp', '88.8 bhp', '177 bhp',\n",
       "       '136 bhp', '201.15 bhp', '93.7 bhp', '177.6 bhp', '313 bhp',\n",
       "       '245 bhp', '125 bhp', '141 bhp', '227 bhp', '62 bhp', '141.1 bhp',\n",
       "       '83.14 bhp', '192 bhp', '67.05 bhp', '47 bhp', '235 bhp', '37 bhp',\n",
       "       '87.2 bhp', '203 bhp', '204 bhp', '246.74 bhp', '122 bhp',\n",
       "       '282 bhp', '181 bhp', '224 bhp', '94 bhp', '367 bhp', '98.79 bhp',\n",
       "       '62.1 bhp', '174.3 bhp', '114 bhp', '335.2 bhp', '169 bhp',\n",
       "       '191.34 bhp', '108.49 bhp', '138.02 bhp', '156 bhp', '187.4 bhp',\n",
       "       '66 bhp', '103.3 bhp', '164.7 bhp', '79.4 bhp', '198.5 bhp',\n",
       "       '154 bhp', '73.8 bhp', '181.43 bhp', '85.8 bhp', '207.8 bhp',\n",
       "       '108.4 bhp', '88 bhp', '63 bhp', '82.5 bhp', '364.9 bhp',\n",
       "       '107.2 bhp', '113.98 bhp', '126.3 bhp', '185 bhp', '237.4 bhp',\n",
       "       '99.6 bhp', '66.7 bhp', '160 bhp', '306 bhp', '98.59 bhp',\n",
       "       '92.7 bhp', '147.51 bhp', '197.2 bhp', '167.6 bhp', '165 bhp',\n",
       "       '110.4 bhp', '73.97 bhp', '147.9 bhp', '116.6 bhp', '148 bhp',\n",
       "       '34.2 bhp', '155 bhp', '197 bhp', '108.62 bhp', '118.3 bhp',\n",
       "       '38.4 bhp', '241.38 bhp', '153.86 bhp', '163.5 bhp', '226.6 bhp',\n",
       "       '84.8 bhp', '53.64 bhp', '158.2 bhp', '69.01 bhp', '181.03 bhp',\n",
       "       '58.2 bhp', '104.68 bhp', '126.24 bhp', '73.75 bhp', '158 bhp',\n",
       "       '130.2 bhp', '57.5 bhp', '97.7 bhp', '121.4 bhp', '98.96 bhp',\n",
       "       '174.5 bhp', '308 bhp', '121.36 bhp', '138 bhp', '265 bhp',\n",
       "       '84 bhp', '321 bhp', '91.72 bhp', '65.3 bhp', '88.2 bhp', '93 bhp',\n",
       "       '35.5 bhp', '86.79 bhp', '157.7 bhp', '40.3 bhp', '91.7 bhp',\n",
       "       '180 bhp', '114.4 bhp', '158.8 bhp', '157.8 bhp', '123.7 bhp',\n",
       "       '56.3 bhp', '189 bhp', '104 bhp', '210 bhp', '270.88 bhp',\n",
       "       '142 bhp', '255 bhp', '236 bhp', '167.7 bhp', '148.31 bhp',\n",
       "       '80.46 bhp', '138.08 bhp', '250 bhp', '74.9 bhp', '91.2 bhp',\n",
       "       '102.57 bhp', '97.6 bhp', '102.53 bhp', '240 bhp', '254 bhp',\n",
       "       '112.4 bhp', '73.74 bhp', '108.495 bhp', '116.9 bhp', '101 bhp',\n",
       "       '320 bhp', '70.02 bhp', '261.49 bhp', '105.5 bhp', '550 bhp',\n",
       "       '168.7 bhp', '55.23 bhp', '94.68 bhp', '152.88 bhp', '163.2 bhp',\n",
       "       '203.2 bhp', '241 bhp', '95 bhp', '200 bhp', '271.23 bhp',\n",
       "       '63.12 bhp', '85.7 bhp', '308.43 bhp', '118.6 bhp', '199.3 bhp',\n",
       "       '83.83 bhp', '55 bhp', '83 bhp', '300 bhp', '201 bhp', '262.6 bhp',\n",
       "       '163 bhp', '58.33 bhp', '86.76 bhp', '76.9 bhp', '174.57 bhp',\n",
       "       '301.73 bhp', '68.1 bhp', '162 bhp', '394.3 bhp', '80.9 bhp',\n",
       "       '147.5 bhp', '272 bhp', '340 bhp', '120.33 bhp', '82.4 bhp',\n",
       "       '231.1 bhp', '335.3 bhp', '333 bhp', '198.25 bhp', '224.34 bhp',\n",
       "       '402 bhp', '261 bhp', '61 bhp', '144 bhp', '71.01 bhp',\n",
       "       '271.72 bhp', '134 bhp', '135.1 bhp', '92 bhp', '64.08 bhp',\n",
       "       '261.5 bhp', '123.37 bhp', '175.67 bhp', '53 bhp', '110.5 bhp',\n",
       "       '178.4 bhp', '193.1 bhp', '41 bhp', '395 bhp', '48.21 bhp',\n",
       "       '450 bhp', '421 bhp', '89.75 bhp', '387.3 bhp', '130.3 bhp',\n",
       "       '281.61 bhp', '52.8 bhp', '139.01 bhp', '208 bhp', '503 bhp',\n",
       "       '168 bhp', '98.82 bhp', '139.07 bhp', '83.11 bhp', '74.93 bhp',\n",
       "       '382 bhp', '74.96 bhp', '552 bhp', '127 bhp', '560 bhp',\n",
       "       '116.4 bhp', '161.6 bhp', '488.1 bhp', '103 bhp', '181.04 bhp'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Seats.unique()\n",
    "df.Power.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Mileage']=df['Mileage'].str.extract('(\\d+)')\n",
    "df['Engine']=df['Engine'].str.extract('(\\d+)')\n",
    "df['Power']=df['Power'].str.extract('(\\d+)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 6019 entries, 0 to 6018\n",
      "Data columns (total 13 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Name               6019 non-null   object \n",
      " 1   Location           6019 non-null   object \n",
      " 2   Year               6019 non-null   int64  \n",
      " 3   Kilometers_Driven  6019 non-null   int64  \n",
      " 4   Fuel_Type          6019 non-null   object \n",
      " 5   Transmission       6019 non-null   object \n",
      " 6   Owner_Type         6019 non-null   object \n",
      " 7   Mileage            6017 non-null   object \n",
      " 8   Engine             5983 non-null   object \n",
      " 9   Power              5876 non-null   object \n",
      " 10  Seats              5977 non-null   float64\n",
      " 11  New_Price          824 non-null    object \n",
      " 12  Price              6019 non-null   float64\n",
      "dtypes: float64(2), int64(2), object(9)\n",
      "memory usage: 658.3+ KB\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0        1.75\n",
       "1       12.50\n",
       "2        4.50\n",
       "3        6.00\n",
       "4       17.74\n",
       "        ...  \n",
       "6014     4.75\n",
       "6015     4.00\n",
       "6016     2.90\n",
       "6017     2.65\n",
       "6018     2.50\n",
       "Name: Price, Length: 6019, dtype: float64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.info()\n",
    "df.Price"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['First', 'Second', 'Fourth & Above', 'Third'], dtype=object)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Owner_Type.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'Name': {'Ambassador Classic Nova Diesel': 0, 'Audi A3 35 TDI Attraction': 1, 'Audi A3 35 TDI Premium': 2, 'Audi A3 35 TDI Premium Plus': 3, 'Audi A3 35 TDI Technology': 4, 'Audi A4 1.8 TFSI': 5, 'Audi A4 1.8 TFSI Technology Edition': 6, 'Audi A4 2.0 TDI': 7, 'Audi A4 2.0 TDI 177 Bhp Premium Plus': 8, 'Audi A4 2.0 TDI 177 Bhp Technology Edition': 9, 'Audi A4 2.0 TDI Celebration Edition': 10, 'Audi A4 2.0 TDI Multitronic': 11, 'Audi A4 2.0 TDI Premium Sport Limited Edition': 12, 'Audi A4 2.0 TFSI': 13, 'Audi A4 3.0 TDI Quattro': 14, 'Audi A4 3.0 TDI Quattro Premium': 15, 'Audi A4 3.2 FSI Tiptronic Quattro': 16, 'Audi A4 30 TFSI Premium Plus': 17, 'Audi A4 35 TDI Premium': 18, 'Audi A4 35 TDI Premium Plus': 19, 'Audi A4 35 TDI Premium Sport': 20, 'Audi A4 35 TDI Technology': 21, 'Audi A4 35 TDI Technology Edition': 22, 'Audi A4 New 2.0 TDI Multitronic': 23, 'Audi A6 2.0 TDI Design Edition': 24, 'Audi A6 2.7 TDI': 25, 'Audi A6 2.8 FSI': 26, 'Audi A6 2011-2015 2.0 TDI': 27, 'Audi A6 2011-2015 2.0 TDI Premium Plus': 28, 'Audi A6 2011-2015 2.0 TDI Technology': 29, 'Audi A6 2011-2015 3.0 TDI Quattro Premium Plus': 30, 'Audi A6 2011-2015 35 TDI Premium': 31, 'Audi A6 2011-2015 35 TDI Technology': 32, 'Audi A6 2011-2015 35 TFSI Technology': 33, 'Audi A6 3.0 TDI quattro': 34, 'Audi A6 35 TDI Matrix': 35, 'Audi A6 35 TFSI Matrix': 36, 'Audi A7 2011-2015 3.0 TDI Quattro': 37, 'Audi A7 2011-2015 Sportback': 38, 'Audi A8 L 3.0 TDI quattro': 39, 'Audi Q3 2.0 TDI': 40, 'Audi Q3 2.0 TDI Quattro': 41, 'Audi Q3 2012-2015 2.0 TDI Quattro Premium Plus': 42, 'Audi Q3 2012-2015 35 TDI Quattro Premium': 43, 'Audi Q3 2012-2015 35 TDI Quattro Premium Plus': 44, 'Audi Q3 30 TDI Premium FWD': 45, 'Audi Q3 35 TDI Quattro Premium': 46, 'Audi Q3 35 TDI Quattro Premium Plus': 47, 'Audi Q5 2.0 TDI': 48, 'Audi Q5 2.0 TDI Premium Plus': 49, 'Audi Q5 2.0 TDI Technology': 50, 'Audi Q5 2008-2012 2.0 TDI': 51, 'Audi Q5 2008-2012 2.0 TFSI Quattro': 52, 'Audi Q5 2008-2012 3.0 TDI': 53, 'Audi Q5 3.0 TDI Quattro': 54, 'Audi Q5 3.0 TDI Quattro Technology': 55, 'Audi Q5 30 TDI quattro Premium': 56, 'Audi Q5 30 TDI quattro Premium Plus': 57, 'Audi Q7 3.0 TDI Quattro Premium Plus': 58, 'Audi Q7 3.0 TDI Quattro Technology': 59, 'Audi Q7 3.0 TDI quattro': 60, 'Audi Q7 35 TDI Quattro Premium Plus': 61, 'Audi Q7 35 TDI Quattro Technology': 62, 'Audi Q7 4.2 FSI quattro': 63, 'Audi Q7 4.2 TDI Quattro Technology': 64, 'Audi Q7 45 TDI Quattro Premium Plus': 65, 'Audi Q7 45 TDI Quattro Technology': 66, 'Audi RS5 Coupe': 67, 'Audi TT 2.0 TFSI': 68, 'Audi TT 40 TFSI': 69, 'BMW 1 Series 118d Base': 70, 'BMW 1 Series 118d Sport Line': 71, 'BMW 3 Series 2005-2011 325i Sedan': 72, 'BMW 3 Series 318i Sedan': 73, 'BMW 3 Series 320d': 74, 'BMW 3 Series 320d Corporate Edition': 75, 'BMW 3 Series 320d Dynamic': 76, 'BMW 3 Series 320d GT Luxury Line': 77, 'BMW 3 Series 320d Highline': 78, 'BMW 3 Series 320d Luxury Line': 79, 'BMW 3 Series 320d Luxury Plus': 80, 'BMW 3 Series 320d M Sport': 81, 'BMW 3 Series 320d Prestige': 82, 'BMW 3 Series 320d Sedan': 83, 'BMW 3 Series 320d Sport': 84, 'BMW 3 Series 320d Sport Line': 85, 'BMW 3 Series 320i': 86, 'BMW 3 Series 320i Sedan': 87, 'BMW 3 Series 328i Sport Line': 88, 'BMW 3 Series 330 Ci Convertible': 89, 'BMW 3 Series 330d Convertible': 90, 'BMW 3 Series 330i': 91, 'BMW 3 Series GT 320d Luxury Line': 92, 'BMW 3 Series Luxury Line': 93, 'BMW 3 Series Sport': 94, 'BMW 5 Series 2003-2012 520d': 95, 'BMW 5 Series 2003-2012 523i': 96, 'BMW 5 Series 2003-2012 525d': 97, 'BMW 5 Series 2003-2012 530d': 98, 'BMW 5 Series 2003-2012 530d Highline': 99, 'BMW 5 Series 2003-2012 540i Sedan': 100, 'BMW 5 Series 2003-2012 GT 530d LE': 101, 'BMW 5 Series 2007-2010 525d Sedan': 102, 'BMW 5 Series 2010-2013 525i': 103, 'BMW 5 Series 2010-2013 530d': 104, 'BMW 5 Series 2013-2017 520d Luxury Line': 105, 'BMW 5 Series 2013-2017 520d M Sport': 106, 'BMW 5 Series 2013-2017 525d Luxury Line': 107, 'BMW 5 Series 2013-2017 530d M Sport': 108, 'BMW 5 Series 520d Luxury Line': 109, 'BMW 5 Series 520d Sedan': 110, 'BMW 5 Series 520d Sport Line': 111, 'BMW 5 Series 523i Sedan': 112, 'BMW 5 Series 525d Sedan': 113, 'BMW 5 Series 530d Highline Sedan': 114, 'BMW 5 Series 530d M Sport': 115, 'BMW 5 Series 530i Sedan': 116, 'BMW 6 Series 630i Coupe': 117, 'BMW 6 Series 640d Coupe': 118, 'BMW 6 Series 640d Gran Coupe': 119, 'BMW 6 Series 650i Coupe': 120, 'BMW 6 Series Gran Coupe': 121, 'BMW 7 Series 2007-2012 730Ld': 122, 'BMW 7 Series 2007-2012 740Li': 123, 'BMW 7 Series 2007-2012 750Li': 124, 'BMW 7 Series 730Ld': 125, 'BMW 7 Series 730Ld Design Pure Excellence CBU': 126, 'BMW 7 Series 730Ld Eminence': 127, 'BMW 7 Series 730Ld Sedan': 128, 'BMW 7 Series 740Li': 129, 'BMW X1 M Sport sDrive 20d': 130, 'BMW X1 sDrive 18i': 131, 'BMW X1 sDrive 20d Exclusive': 132, 'BMW X1 sDrive 20d Sportline': 133, 'BMW X1 sDrive 20d xLine': 134, 'BMW X1 sDrive20d': 135, 'BMW X1 xDrive 20d M Sport': 136, 'BMW X1 xDrive 20d xLine': 137, 'BMW X3 xDrive 20d Expedition': 138, 'BMW X3 xDrive 20d Luxury Line': 139, 'BMW X3 xDrive 20d xLine': 140, 'BMW X3 xDrive20d': 141, 'BMW X3 xDrive20d Advantage Edition': 142, 'BMW X3 xDrive20d Expedition': 143, 'BMW X3 xDrive20d xLine': 144, 'BMW X3 xDrive30d M Sport': 145, 'BMW X5 2014-2019 xDrive 30d Design Pure Experience 7 Seater': 146, 'BMW X5 3.0d': 147, 'BMW X5 X5 M': 148, 'BMW X5 xDrive 30d': 149, 'BMW X5 xDrive 30d Design Pure Experience 5 Seater': 150, 'BMW X5 xDrive 30d M Sport': 151, 'BMW X6 xDrive 40d': 152, 'BMW X6 xDrive 40d M Sport': 153, 'BMW X6 xDrive30d': 154, 'BMW Z4 2009-2013 35i': 155, 'BMW Z4 2009-2013 Roadster 2.5i': 156, 'Bentley Continental Flying Spur': 157, 'Chevrolet Aveo 1.4': 158, 'Chevrolet Aveo 1.4 LS': 159, 'Chevrolet Aveo 1.4 LT': 160, 'Chevrolet Aveo 1.4 LT ABS BSIV': 161, 'Chevrolet Aveo 1.6 LT': 162, 'Chevrolet Aveo U-VA 1.2 LS': 163, 'Chevrolet Aveo U-VA 1.2 LT': 164, 'Chevrolet Beat Diesel': 165, 'Chevrolet Beat Diesel LS': 166, 'Chevrolet Beat Diesel LT': 167, 'Chevrolet Beat Diesel PS': 168, 'Chevrolet Beat LS': 169, 'Chevrolet Beat LT': 170, 'Chevrolet Beat LT Option': 171, 'Chevrolet Beat Option Pack': 172, 'Chevrolet Captiva LT': 173, 'Chevrolet Captiva LTZ VCDi': 174, 'Chevrolet Cruze LTZ': 175, 'Chevrolet Cruze LTZ AT': 176, 'Chevrolet Enjoy 1.3 TCDi LS 8': 177, 'Chevrolet Enjoy 1.3 TCDi LTZ 8': 178, 'Chevrolet Enjoy 1.4 LS 8': 179, 'Chevrolet Enjoy Petrol LS 7 Seater': 180, 'Chevrolet Enjoy TCDi LS 8 Seater': 181, 'Chevrolet Enjoy TCDi LTZ 7 Seater': 182, 'Chevrolet Optra 1.6 Elite': 183, 'Chevrolet Optra 1.6 LS': 184, 'Chevrolet Optra 1.6 LT Royale': 185, 'Chevrolet Optra Magnum 1.6 LS BS3': 186, 'Chevrolet Optra Magnum 1.6 LS Petrol': 187, 'Chevrolet Optra Magnum 1.6 LT ABS BS3': 188, 'Chevrolet Optra Magnum 2.0 LS BSIII': 189, 'Chevrolet Optra Magnum 2.0 LT': 190, 'Chevrolet Optra Magnum 2.0 LT BS3': 191, 'Chevrolet Sail 1.2 LS': 192, 'Chevrolet Sail Hatchback 1.2 LS': 193, 'Chevrolet Sail Hatchback 1.2 LS ABS': 194, 'Chevrolet Sail Hatchback LS ABS': 195, 'Chevrolet Sail LT ABS': 196, 'Chevrolet Spark 1.0 LS': 197, 'Chevrolet Spark 1.0 LT': 198, 'Chevrolet Tavera LS B3 10 Seats BSIII': 199, 'Chevrolet Tavera LT 9 Str BS IV': 200, 'Datsun GO NXT': 201, 'Datsun GO Plus A': 202, 'Datsun GO Plus T': 203, 'Datsun GO Plus T Petrol': 204, 'Datsun GO T Option': 205, 'Datsun Redi GO Sport': 206, 'Datsun redi-GO S': 207, 'Datsun redi-GO T': 208, 'Datsun redi-GO T Option': 209, 'Fiat Avventura MULTIJET Emotion': 210, 'Fiat Grande Punto 1.2 Dynamic': 211, 'Fiat Grande Punto 1.3 Emotion (Diesel)': 212, 'Fiat Grande Punto 1.3 Emotion Pack (Diesel)': 213, 'Fiat Grande Punto 1.4 Emotion': 214, 'Fiat Grande Punto EVO 1.3 Active': 215, 'Fiat Linea 1.3 Dynamic': 216, 'Fiat Linea 1.3 Emotion': 217, 'Fiat Linea Classic Plus 1.3 Multijet': 218, 'Fiat Linea Emotion': 219, 'Fiat Linea Emotion (Diesel)': 220, 'Fiat Linea Emotion Pack (Diesel)': 221, 'Fiat Linea T Jet': 222, 'Fiat Linea T-Jet Active': 223, 'Fiat Petra 1.2 EL': 224, 'Fiat Punto 1.2 Dynamic': 225, 'Fiat Punto 1.3 Emotion': 226, 'Fiat Punto 1.4 Emotion': 227, 'Fiat Punto EVO 1.2 Emotion': 228, 'Fiat Siena 1.2 ELX': 229, 'Force One LX 4x4': 230, 'Force One LX ABS 7 Seating': 231, 'Ford Aspire Ambiente Diesel': 232, 'Ford Aspire Titanium Diesel': 233, 'Ford Aspire Titanium Plus Diesel': 234, 'Ford Classic 1.4 Duratorq Titanium': 235, 'Ford EcoSport 1.0 Ecoboost Titanium Plus': 236, 'Ford EcoSport 1.0 Ecoboost Titanium Plus BE': 237, 'Ford EcoSport 1.5 Diesel Titanium': 238, 'Ford EcoSport 1.5 Diesel Titanium Plus': 239, 'Ford EcoSport 1.5 Diesel Trend': 240, 'Ford EcoSport 1.5 Petrol Titanium': 241, 'Ford EcoSport 1.5 Petrol Titanium Plus AT': 242, 'Ford EcoSport 1.5 Petrol Trend': 243, 'Ford EcoSport 1.5 TDCi Ambiente': 244, 'Ford EcoSport 1.5 TDCi Titanium': 245, 'Ford EcoSport 1.5 TDCi Trend': 246, 'Ford EcoSport 1.5 TDCi Trend Plus': 247, 'Ford EcoSport 1.5 TDCi Trend Plus BE': 248, 'Ford EcoSport 1.5 Ti VCT AT Titanium': 249, 'Ford EcoSport 1.5 Ti VCT AT Titanium BE': 250, 'Ford EcoSport 1.5 Ti VCT MT Signature': 251, 'Ford EcoSport 1.5 Ti VCT MT Titanium': 252, 'Ford EcoSport 1.5 Ti VCT MT Titanium BE': 253, 'Ford Ecosport 1.0 Ecoboost Platinum Edition': 254, 'Ford Ecosport 1.0 Ecoboost Titanium': 255, 'Ford Ecosport 1.5 DV5 MT Ambiente': 256, 'Ford Ecosport 1.5 DV5 MT Titanium': 257, 'Ford Ecosport 1.5 DV5 MT Titanium Optional': 258, 'Ford Ecosport 1.5 DV5 MT Trend': 259, 'Ford Ecosport 1.5 Ti VCT AT Titanium': 260, 'Ford Ecosport 1.5 Ti VCT MT Titanium': 261, 'Ford Ecosport 1.5 Ti VCT MT Trend': 262, 'Ford Ecosport Signature Edition Diesel': 263, 'Ford Endeavour 2.2 Titanium AT 4X2': 264, 'Ford Endeavour 2.2 Trend AT 4X2': 265, 'Ford Endeavour 2.2 Trend MT 4X4': 266, 'Ford Endeavour 2.5L 4X2': 267, 'Ford Endeavour 2.5L 4X2 MT': 268, 'Ford Endeavour 3.0L 4X4 AT': 269, 'Ford Endeavour 3.2 Titanium AT 4X4': 270, 'Ford Endeavour 3.2 Trend AT 4X4': 271, 'Ford Endeavour 4x2 XLT': 272, 'Ford Endeavour 4x2 XLT Limited Edition': 273, 'Ford Endeavour Hurricane LE': 274, 'Ford Endeavour Hurricane Limited Edition': 275, 'Ford Endeavour Titanium 4X2': 276, 'Ford Endeavour XLT TDCi 4X2': 277, 'Ford Endeavour XLT TDCi 4X4': 278, 'Ford Fiesta 1.4 Duratec EXI': 279, 'Ford Fiesta 1.4 Duratec EXI Limited Edition': 280, 'Ford Fiesta 1.4 Duratec ZXI': 281, 'Ford Fiesta 1.4 Duratorq EXI': 282, 'Ford Fiesta 1.4 Duratorq ZXI': 283, 'Ford Fiesta 1.4 SXi TDCi': 284, 'Ford Fiesta 1.4 SXi TDCi ABS': 285, 'Ford Fiesta 1.4 TDCi EXI': 286, 'Ford Fiesta 1.4 TDCi EXI Limited Edition': 287, 'Ford Fiesta 1.4 ZXi Duratec': 288, 'Ford Fiesta 1.4 ZXi Leather': 289, 'Ford Fiesta 1.4 ZXi TDCi ABS': 290, 'Ford Fiesta 1.5 TDCi Titanium': 291, 'Ford Fiesta 1.6 SXI ABS Duratec': 292, 'Ford Fiesta 1.6 ZXI Duratec': 293, 'Ford Fiesta 1.6 ZXi ABS': 294, 'Ford Fiesta 1.6 ZXi Duratec': 295, 'Ford Fiesta 1.6 ZXi Leather': 296, 'Ford Fiesta Classic 1.4 Duratorq CLXI': 297, 'Ford Fiesta Classic 1.4 Duratorq LXI': 298, 'Ford Fiesta Diesel Style': 299, 'Ford Fiesta Diesel Titanium Plus': 300, 'Ford Fiesta Diesel Trend': 301, 'Ford Fiesta EXi 1.4 TDCi Ltd': 302, 'Ford Fiesta Titanium 1.5 TDCi': 303, 'Ford Figo 1.2P Titanium MT': 304, 'Ford Figo 1.5D Titanium Plus MT': 305, 'Ford Figo 1.5D Trend MT': 306, 'Ford Figo 2015-2019 1.2P Sports Edition MT': 307, 'Ford Figo 2015-2019 1.2P Titanium MT': 308, 'Ford Figo 2015-2019 1.2P Titanium Opt MT': 309, 'Ford Figo 2015-2019 1.2P Trend MT': 310, 'Ford Figo 2015-2019 1.5D Ambiente MT': 311, 'Ford Figo 2015-2019 1.5D Titanium MT': 312, 'Ford Figo 2015-2019 1.5D Titanium Plus MT': 313, 'Ford Figo 2015-2019 1.5D Trend MT': 314, 'Ford Figo 2015-2019 1.5P Titanium AT': 315, 'Ford Figo Aspire 1.2 Ti-VCT Ambiente': 316, 'Ford Figo Aspire 1.2 Ti-VCT Titanium': 317, 'Ford Figo Aspire 1.2 Ti-VCT Trend': 318, 'Ford Figo Aspire 1.5 TDCi Ambiente': 319, 'Ford Figo Aspire 1.5 TDCi Titanium': 320, 'Ford Figo Aspire 1.5 TDCi Titanium Plus': 321, 'Ford Figo Aspire 1.5 TDCi Trend': 322, 'Ford Figo Aspire 1.5 Ti-VCT Titanium': 323, 'Ford Figo Diesel': 324, 'Ford Figo Diesel Celebration Edition': 325, 'Ford Figo Diesel EXI': 326, 'Ford Figo Diesel EXI Option': 327, 'Ford Figo Diesel LXI': 328, 'Ford Figo Diesel Titanium': 329, 'Ford Figo Diesel ZXI': 330, 'Ford Figo Petrol EXI': 331, 'Ford Figo Petrol LXI': 332, 'Ford Figo Petrol Titanium': 333, 'Ford Figo Petrol ZXI': 334, 'Ford Figo Titanium Diesel': 335, 'Ford Freestyle Titanium Petrol': 336, 'Ford Freestyle Titanium Plus Petrol': 337, 'Ford Fusion Plus 1.4 TDCi Diesel': 338, 'Ford Ikon 1.3 CLXi': 339, 'Ford Ikon 1.3 Flair': 340, 'Ford Ikon 1.3 LXi NXt': 341, 'Ford Ikon 1.4 TDCi DuraTorq': 342, 'Ford Ikon 1.6 CLXI': 343, 'Ford Ikon 1.6 ZXI NXt': 344, 'Ford Mustang V8': 345, 'Honda Accord 2.4 A/T': 346, 'Honda Accord 2.4 AT': 347, 'Honda Accord 2.4 Elegance A/T': 348, 'Honda Accord 2.4 Elegance M/T': 349, 'Honda Accord 2.4 Inspire M/T': 350, 'Honda Accord 2.4 M/T': 351, 'Honda Accord 2.4 MT': 352, 'Honda Accord 2001-2003 2.0 AT': 353, 'Honda Accord 2001-2003 2.3 VTi L AT': 354, 'Honda Accord V6 AT': 355, 'Honda Accord VTi-L (AT)': 356, 'Honda Accord VTi-L AT': 357, 'Honda Accord VTi-L MT': 358, 'Honda Amaze E i-Dtech': 359, 'Honda Amaze E i-Vtech': 360, 'Honda Amaze EX i-Dtech': 361, 'Honda Amaze EX i-Vtech': 362, 'Honda Amaze S AT i-Vtech': 363, 'Honda Amaze S Diesel': 364, 'Honda Amaze S Petrol': 365, 'Honda Amaze S i-DTEC': 366, 'Honda Amaze S i-Dtech': 367, 'Honda Amaze S i-VTEC': 368, 'Honda Amaze S i-Vtech': 369, 'Honda Amaze SX i-DTEC': 370, 'Honda Amaze SX i-VTEC': 371, 'Honda Amaze V CVT Petrol': 372, 'Honda Amaze V Petrol': 373, 'Honda Amaze VX AT i-Vtech': 374, 'Honda Amaze VX Diesel': 375, 'Honda Amaze VX O i VTEC': 376, 'Honda Amaze VX Petrol': 377, 'Honda Amaze VX i-DTEC': 378, 'Honda Amaze VX i-VTEC': 379, 'Honda Amaze VX i-Vtech': 380, 'Honda BR-V i-DTEC VX MT': 381, 'Honda BR-V i-VTEC S MT': 382, 'Honda BRV i-VTEC V CVT': 383, 'Honda BRV i-VTEC V MT': 384, 'Honda Brio 1.2 S MT': 385, 'Honda Brio 1.2 S Option MT': 386, 'Honda Brio 1.2 VX AT': 387, 'Honda Brio 1.2 VX MT': 388, 'Honda Brio E MT': 389, 'Honda Brio EX MT': 390, 'Honda Brio S MT': 391, 'Honda Brio S Option AT': 392, 'Honda Brio S Option MT': 393, 'Honda Brio V MT': 394, 'Honda Brio VX': 395, 'Honda Brio VX AT': 396, 'Honda CR-V 2.0 AT': 397, 'Honda CR-V 2.0L 2WD AT': 398, 'Honda CR-V 2.0L 2WD MT': 399, 'Honda CR-V 2.4 4WD AT': 400, 'Honda CR-V 2.4 AT': 401, 'Honda CR-V 2.4 MT': 402, 'Honda CR-V 2.4L 4WD AT': 403, 'Honda CR-V 2.4L 4WD AT AVN': 404, 'Honda CR-V 2.4L 4WD MT': 405, 'Honda CR-V AT With Sun Roof': 406, 'Honda CR-V Petrol 2WD': 407, 'Honda CR-V RVi MT': 408, 'Honda CR-V Sport': 409, 'Honda City 1.3 DX': 410, 'Honda City 1.3 EXI': 411, 'Honda City 1.3 EXI S': 412, 'Honda City 1.5 E MT': 413, 'Honda City 1.5 EXI': 414, 'Honda City 1.5 EXI AT': 415, 'Honda City 1.5 EXI S': 416, 'Honda City 1.5 GXI': 417, 'Honda City 1.5 S AT': 418, 'Honda City 1.5 S MT': 419, 'Honda City 1.5 V AT': 420, 'Honda City 1.5 V AT Exclusive': 421, 'Honda City 1.5 V AT Sunroof': 422, 'Honda City 1.5 V MT': 423, 'Honda City 1.5 V MT Exclusive': 424, 'Honda City 1.5 V MT Sunroof': 425, 'Honda City Corporate Edition': 426, 'Honda City V AT': 427, 'Honda City V MT': 428, 'Honda City V MT AVN': 429, 'Honda City V MT Exclusive': 430, 'Honda City ZX CVT': 431, 'Honda City ZX EXi': 432, 'Honda City ZX GXi': 433, 'Honda City ZX VTEC': 434, 'Honda City i DTEC S': 435, 'Honda City i DTEC SV': 436, 'Honda City i DTEC V': 437, 'Honda City i DTEC VX': 438, 'Honda City i DTEC VX Option': 439, 'Honda City i DTec E': 440, 'Honda City i DTec S': 441, 'Honda City i DTec SV': 442, 'Honda City i DTec V': 443, 'Honda City i DTec VX': 444, 'Honda City i VTEC CVT SV': 445, 'Honda City i VTEC CVT VX': 446, 'Honda City i VTEC E': 447, 'Honda City i VTEC S': 448, 'Honda City i VTEC SV': 449, 'Honda City i VTEC V': 450, 'Honda City i VTEC VX': 451, 'Honda City i VTEC VX Option': 452, 'Honda City i-DTEC SV': 453, 'Honda City i-DTEC ZX': 454, 'Honda City i-VTEC CVT V': 455, 'Honda City i-VTEC CVT VX': 456, 'Honda City i-VTEC CVT ZX': 457, 'Honda City i-VTEC S': 458, 'Honda City i-VTEC V': 459, 'Honda City i-VTEC VX': 460, 'Honda Civic 2006-2010 1.8 (E) MT': 461, 'Honda Civic 2006-2010 1.8 MT Sport': 462, 'Honda Civic 2006-2010 1.8 S AT': 463, 'Honda Civic 2006-2010 1.8 S MT': 464, 'Honda Civic 2006-2010 1.8 V AT': 465, 'Honda Civic 2006-2010 1.8 V MT': 466, 'Honda Civic 2010-2013 1.8 S MT': 467, 'Honda Civic 2010-2013 1.8 V AT Sunroof': 468, 'Honda Civic 2010-2013 1.8 V MT': 469, 'Honda Jazz 1.2 E i VTEC': 470, 'Honda Jazz 1.2 S i VTEC': 471, 'Honda Jazz 1.2 SV i VTEC': 472, 'Honda Jazz 1.2 V AT i VTEC Privilege': 473, 'Honda Jazz 1.2 V CVT i VTEC': 474, 'Honda Jazz 1.2 V i VTEC': 475, 'Honda Jazz 1.2 VX i VTEC': 476, 'Honda Jazz 1.5 S i DTEC': 477, 'Honda Jazz 1.5 SV i DTEC': 478, 'Honda Jazz 1.5 V i DTEC': 479, 'Honda Jazz 1.5 VX i DTEC': 480, 'Honda Jazz Active': 481, 'Honda Jazz Exclusive CVT': 482, 'Honda Jazz Mode': 483, 'Honda Jazz S': 484, 'Honda Jazz Select Edition': 485, 'Honda Jazz V': 486, 'Honda Jazz V CVT': 487, 'Honda Jazz VX': 488, 'Honda Jazz VX Diesel': 489, 'Honda Mobilio E i DTEC': 490, 'Honda Mobilio E i VTEC': 491, 'Honda Mobilio RS Option i DTEC': 492, 'Honda Mobilio RS i DTEC': 493, 'Honda Mobilio S i DTEC': 494, 'Honda Mobilio S i VTEC': 495, 'Honda Mobilio V Option i DTEC': 496, 'Honda Mobilio V i DTEC': 497, 'Honda WR-V Edge Edition i-VTEC S': 498, 'Honda WRV i-VTEC VX': 499, 'Hyundai Accent CRDi': 500, 'Hyundai Accent Executive': 501, 'Hyundai Accent Executive CNG': 502, 'Hyundai Accent GLE': 503, 'Hyundai Accent GLE 1': 504, 'Hyundai Accent GLS': 505, 'Hyundai Accent GLS 1.6': 506, 'Hyundai Creta 1.4 CRDi S': 507, 'Hyundai Creta 1.4 CRDi S Plus': 508, 'Hyundai Creta 1.4 E Plus Diesel': 509, 'Hyundai Creta 1.6 CRDi AT SX Plus': 510, 'Hyundai Creta 1.6 CRDi SX': 511, 'Hyundai Creta 1.6 CRDi SX Option': 512, 'Hyundai Creta 1.6 SX': 513, 'Hyundai Creta 1.6 SX Automatic Diesel': 514, 'Hyundai Creta 1.6 SX Option': 515, 'Hyundai Creta 1.6 SX Option Diesel': 516, 'Hyundai Creta 1.6 SX Option Executive': 517, 'Hyundai Creta 1.6 SX Plus Diesel': 518, 'Hyundai Creta 1.6 SX Plus Dual Tone Petrol': 519, 'Hyundai Creta 1.6 SX Plus Petrol Automatic': 520, 'Hyundai Creta 1.6 VTVT E Plus': 521, 'Hyundai Creta 1.6 VTVT S': 522, 'Hyundai EON 1.0 Kappa Magna Plus Optional': 523, 'Hyundai EON 1.0 Magna Plus Option O': 524, 'Hyundai EON D Lite': 525, 'Hyundai EON D Lite Plus': 526, 'Hyundai EON D Lite Plus Option': 527, 'Hyundai EON Era': 528, 'Hyundai EON Era Plus': 529, 'Hyundai EON LPG Era Plus Option': 530, 'Hyundai EON Magna': 531, 'Hyundai EON Magna Plus': 532, 'Hyundai EON Sportz': 533, 'Hyundai Elantra 1.6 SX': 534, 'Hyundai Elantra 1.6 SX Option AT': 535, 'Hyundai Elantra 2.0 SX Option AT': 536, 'Hyundai Elantra CRDi': 537, 'Hyundai Elantra CRDi S': 538, 'Hyundai Elantra CRDi SX': 539, 'Hyundai Elantra CRDi SX AT': 540, 'Hyundai Elantra SX': 541, 'Hyundai Elite i20 Asta Option': 542, 'Hyundai Elite i20 Diesel Asta Option': 543, 'Hyundai Elite i20 Petrol Asta': 544, 'Hyundai Elite i20 Petrol Sportz': 545, 'Hyundai Elite i20 Sportz Plus': 546, 'Hyundai Getz 1.3 GLS': 547, 'Hyundai Getz 1.5 CRDi GVS': 548, 'Hyundai Getz GLE': 549, 'Hyundai Getz GLS': 550, 'Hyundai Getz GLS ABS': 551, 'Hyundai Getz GVS': 552, 'Hyundai Grand i10 1.2 CRDi Asta': 553, 'Hyundai Grand i10 1.2 CRDi Magna': 554, 'Hyundai Grand i10 1.2 CRDi Sportz': 555, 'Hyundai Grand i10 1.2 CRDi Sportz Option': 556, 'Hyundai Grand i10 1.2 Kappa Asta': 557, 'Hyundai Grand i10 1.2 Kappa Magna': 558, 'Hyundai Grand i10 1.2 Kappa Magna AT': 559, 'Hyundai Grand i10 1.2 Kappa Sportz': 560, 'Hyundai Grand i10 1.2 Kappa Sportz AT': 561, 'Hyundai Grand i10 1.2 Kappa Sportz Option': 562, 'Hyundai Grand i10 AT Asta': 563, 'Hyundai Grand i10 Asta': 564, 'Hyundai Grand i10 Asta Option': 565, 'Hyundai Grand i10 Asta Option AT': 566, 'Hyundai Grand i10 CRDi Asta': 567, 'Hyundai Grand i10 CRDi Era': 568, 'Hyundai Grand i10 CRDi Magna': 569, 'Hyundai Grand i10 CRDi SportZ Edition': 570, 'Hyundai Grand i10 CRDi Sportz': 571, 'Hyundai Grand i10 CRDi Sportz Celebration Edition': 572, 'Hyundai Grand i10 Era': 573, 'Hyundai Grand i10 Magna': 574, 'Hyundai Grand i10 Magna AT': 575, 'Hyundai Grand i10 SportZ Edition': 576, 'Hyundai Grand i10 Sportz': 577, 'Hyundai Santa Fe 2WD AT': 578, 'Hyundai Santa Fe 4WD AT': 579, 'Hyundai Santa Fe 4X2': 580, 'Hyundai Santa Fe 4X4': 581, 'Hyundai Santa Fe 4x4 AT': 582, 'Hyundai Santro AT': 583, 'Hyundai Santro D Lite': 584, 'Hyundai Santro DX': 585, 'Hyundai Santro GLS I - Euro I': 586, 'Hyundai Santro GLS I - Euro II': 587, 'Hyundai Santro GLS II - Euro II': 588, 'Hyundai Santro GS': 589, 'Hyundai Santro GS zipDrive - Euro II': 590, 'Hyundai Santro LP - Euro II': 591, 'Hyundai Santro LP zipPlus': 592, 'Hyundai Santro LS zipPlus': 593, 'Hyundai Santro Xing GL': 594, 'Hyundai Santro Xing GL Plus': 595, 'Hyundai Santro Xing GL Plus LPG': 596, 'Hyundai Santro Xing GLS': 597, 'Hyundai Santro Xing GLS LPG': 598, 'Hyundai Santro Xing XG': 599, 'Hyundai Santro Xing XG eRLX Euro III': 600, 'Hyundai Santro Xing XL': 601, 'Hyundai Santro Xing XL AT eRLX Euro II': 602, 'Hyundai Santro Xing XL AT eRLX Euro III': 603, 'Hyundai Santro Xing XL eRLX Euro III': 604, 'Hyundai Santro Xing XO': 605, 'Hyundai Santro Xing XO CNG': 606, 'Hyundai Santro Xing XO eRLX Euro II': 607, 'Hyundai Santro Xing XP': 608, 'Hyundai Sonata 2.4 GDI': 609, 'Hyundai Sonata Embera 2.0L CRDi AT': 610, 'Hyundai Sonata Embera 2.0L CRDi MT': 611, 'Hyundai Sonata GOLD': 612, 'Hyundai Sonata Transform 2.4 GDi AT': 613, 'Hyundai Sonata Transform 2.4 GDi MT': 614, 'Hyundai Tucson 2.0 Dual VTVT 2WD AT GL': 615, 'Hyundai Tucson 2.0 e-VGT 2WD AT GLS': 616, 'Hyundai Tucson CRDi': 617, 'Hyundai Verna 1.4 CRDi': 618, 'Hyundai Verna 1.4 CRDi GL': 619, 'Hyundai Verna 1.4 CX VTVT': 620, 'Hyundai Verna 1.4 EX': 621, 'Hyundai Verna 1.4 VTVT': 622, 'Hyundai Verna 1.6 CRDI': 623, 'Hyundai Verna 1.6 CRDI AT SX Option': 624, 'Hyundai Verna 1.6 CRDI SX Option': 625, 'Hyundai Verna 1.6 CRDi AT S': 626, 'Hyundai Verna 1.6 CRDi AT SX': 627, 'Hyundai Verna 1.6 CRDi EX AT': 628, 'Hyundai Verna 1.6 CRDi EX MT': 629, 'Hyundai Verna 1.6 CRDi S': 630, 'Hyundai Verna 1.6 CRDi SX': 631, 'Hyundai Verna 1.6 EX VTVT': 632, 'Hyundai Verna 1.6 SX': 633, 'Hyundai Verna 1.6 SX CRDI (O) AT': 634, 'Hyundai Verna 1.6 SX CRDi (O)': 635, 'Hyundai Verna 1.6 SX VTVT': 636, 'Hyundai Verna 1.6 SX VTVT (O)': 637, 'Hyundai Verna 1.6 SX VTVT (O) AT': 638, 'Hyundai Verna 1.6 SX VTVT AT': 639, 'Hyundai Verna 1.6 VTVT': 640, 'Hyundai Verna 1.6 VTVT AT SX': 641, 'Hyundai Verna 1.6 VTVT EX AT': 642, 'Hyundai Verna 1.6 VTVT S': 643, 'Hyundai Verna 1.6 i ABS': 644, 'Hyundai Verna CRDi': 645, 'Hyundai Verna CRDi 1.4 E': 646, 'Hyundai Verna CRDi 1.6 AT SX Option': 647, 'Hyundai Verna CRDi 1.6 AT SX Plus': 648, 'Hyundai Verna CRDi 1.6 SX': 649, 'Hyundai Verna CRDi 1.6 SX Option': 650, 'Hyundai Verna CRDi ABS': 651, 'Hyundai Verna CRDi SX': 652, 'Hyundai Verna CRDi SX ABS': 653, 'Hyundai Verna SX CRDi AT': 654, 'Hyundai Verna Transform CRDi VGT SX ABS': 655, 'Hyundai Verna Transform SX VGT CRDi': 656, 'Hyundai Verna Transform SX VTVT': 657, 'Hyundai Verna Transform VTVT': 658, 'Hyundai Verna Transform Xxi ABS': 659, 'Hyundai Verna VTVT 1.6 AT EX': 660, 'Hyundai Verna VTVT 1.6 AT SX Option': 661, 'Hyundai Verna VTVT 1.6 AT SX Plus': 662, 'Hyundai Verna VTVT 1.6 SX': 663, 'Hyundai Verna VTVT 1.6 SX Option': 664, 'Hyundai Verna XXi ABS (Petrol)': 665, 'Hyundai Verna Xi (Petrol)': 666, 'Hyundai Xcent 1.1 CRDi Base': 667, 'Hyundai Xcent 1.1 CRDi S': 668, 'Hyundai Xcent 1.1 CRDi S Celebration Edition': 669, 'Hyundai Xcent 1.1 CRDi S Option': 670, 'Hyundai Xcent 1.1 CRDi SX': 671, 'Hyundai Xcent 1.1 CRDi SX Option': 672, 'Hyundai Xcent 1.2 CRDi E Plus': 673, 'Hyundai Xcent 1.2 CRDi S': 674, 'Hyundai Xcent 1.2 Kappa AT S Option': 675, 'Hyundai Xcent 1.2 Kappa AT SX Option': 676, 'Hyundai Xcent 1.2 Kappa Base': 677, 'Hyundai Xcent 1.2 Kappa S': 678, 'Hyundai Xcent 1.2 Kappa S Option': 679, 'Hyundai Xcent 1.2 Kappa S Option CNG': 680, 'Hyundai Xcent 1.2 Kappa SX': 681, 'Hyundai Xcent 1.2 Kappa SX Option': 682, 'Hyundai Xcent 1.2 VTVT E': 683, 'Hyundai Xcent 1.2 VTVT E Plus': 684, 'Hyundai Xcent 1.2 VTVT S': 685, 'Hyundai Xcent 1.2 VTVT SX': 686, 'Hyundai i10 Asta': 687, 'Hyundai i10 Asta 1.2 AT with Sunroof': 688, 'Hyundai i10 Asta AT': 689, 'Hyundai i10 Asta Sunroof AT': 690, 'Hyundai i10 Era': 691, 'Hyundai i10 Era 1.1': 692, 'Hyundai i10 Era 1.1 iTech SE': 693, 'Hyundai i10 Magna': 694, 'Hyundai i10 Magna 1.1': 695, 'Hyundai i10 Magna 1.1 iTech SE': 696, 'Hyundai i10 Magna 1.1L': 697, 'Hyundai i10 Magna 1.2': 698, 'Hyundai i10 Magna 1.2 iTech SE': 699, 'Hyundai i10 Magna AT': 700, 'Hyundai i10 Magna LPG': 701, 'Hyundai i10 Magna Optional 1.1L': 702, 'Hyundai i10 Magna(O) with Sun Roof': 703, 'Hyundai i10 Sportz': 704, 'Hyundai i10 Sportz 1.1L': 705, 'Hyundai i10 Sportz 1.2': 706, 'Hyundai i10 Sportz 1.2 AT': 707, 'Hyundai i10 Sportz AT': 708, 'Hyundai i10 Sportz Option': 709, 'Hyundai i20 1.2 Asta': 710, 'Hyundai i20 1.2 Asta Option': 711, 'Hyundai i20 1.2 Asta with AVN': 712, 'Hyundai i20 1.2 Era': 713, 'Hyundai i20 1.2 Magna': 714, 'Hyundai i20 1.2 Magna Executive': 715, 'Hyundai i20 1.2 Sportz': 716, 'Hyundai i20 1.2 Sportz Option': 717, 'Hyundai i20 1.2 Spotz': 718, 'Hyundai i20 1.4 Asta': 719, 'Hyundai i20 1.4 Asta (AT)': 720, 'Hyundai i20 1.4 Asta CRDi with AVN': 721, 'Hyundai i20 1.4 Asta Option': 722, 'Hyundai i20 1.4 Asta Optional With Sunroof': 723, 'Hyundai i20 1.4 CRDi Asta': 724, 'Hyundai i20 1.4 CRDi Magna': 725, 'Hyundai i20 1.4 CRDi Sportz': 726, 'Hyundai i20 1.4 Magna ABS': 727, 'Hyundai i20 1.4 Sportz': 728, 'Hyundai i20 2015-2017 1.2 Asta': 729, 'Hyundai i20 2015-2017 1.2 Asta with AVN': 730, 'Hyundai i20 2015-2017 1.2 Magna': 731, 'Hyundai i20 2015-2017 1.4 Magna ABS': 732, 'Hyundai i20 2015-2017 Asta': 733, 'Hyundai i20 2015-2017 Asta 1.2': 734, 'Hyundai i20 2015-2017 Magna': 735, 'Hyundai i20 2015-2017 Magna 1.2': 736, 'Hyundai i20 2015-2017 Sportz 1.2': 737, 'Hyundai i20 2015-2017 Sportz AT 1.4': 738, 'Hyundai i20 Active 1.2 S': 739, 'Hyundai i20 Active 1.2 SX': 740, 'Hyundai i20 Active 1.2 SX Dual Tone': 741, 'Hyundai i20 Active 1.4 SX': 742, 'Hyundai i20 Active 1.4 SX Dual Tone': 743, 'Hyundai i20 Active S Diesel': 744, 'Hyundai i20 Active SX Dual Tone Petrol': 745, 'Hyundai i20 Active SX Petrol': 746, 'Hyundai i20 Asta': 747, 'Hyundai i20 Asta (o)': 748, 'Hyundai i20 Asta (o) 1.4 CRDi (Diesel)': 749, 'Hyundai i20 Asta 1.2': 750, 'Hyundai i20 Asta 1.4 CRDi': 751, 'Hyundai i20 Asta 1.4 CRDi (Diesel)': 752, 'Hyundai i20 Asta Option 1.2': 753, 'Hyundai i20 Asta Option 1.4 CRDi': 754, 'Hyundai i20 Asta Optional with Sunroof 1.2': 755, 'Hyundai i20 Diesel Asta Option': 756, 'Hyundai i20 Era 1.4 CRDi': 757, 'Hyundai i20 Magna': 758, 'Hyundai i20 Magna 1.2': 759, 'Hyundai i20 Magna 1.4 CRDi': 760, 'Hyundai i20 Magna 1.4 CRDi (Diesel)': 761, 'Hyundai i20 Magna Optional 1.2': 762, 'Hyundai i20 Magna Optional 1.4 CRDi': 763, 'Hyundai i20 Sportz 1.2': 764, 'Hyundai i20 Sportz 1.4 CRDi': 765, 'Hyundai i20 Sportz AT 1.4': 766, 'Hyundai i20 Sportz Diesel': 767, 'Hyundai i20 Sportz Option': 768, 'Hyundai i20 Sportz Option 1.2': 769, 'Hyundai i20 Sportz Option Diesel': 770, 'Hyundai i20 Sportz Petrol': 771, 'ISUZU D-MAX V-Cross 4X4': 772, 'Isuzu MUX 4WD': 773, 'Jaguar F Type 5.0 V8 S': 774, 'Jaguar XE 2.0L Diesel Prestige': 775, 'Jaguar XE Portfolio': 776, 'Jaguar XF 2.0 Diesel Portfolio': 777, 'Jaguar XF 2.2 Litre Executive': 778, 'Jaguar XF 2.2 Litre Luxury': 779, 'Jaguar XF 3.0 Litre S Premium Luxury': 780, 'Jaguar XF Aero Sport Edition': 781, 'Jaguar XF Diesel': 782, 'Jaguar XJ 2.0L Portfolio': 783, 'Jaguar XJ 3.0L Portfolio': 784, 'Jaguar XJ 3.0L Portfolio LWB': 785, 'Jaguar XJ 3.0L Premium Luxury': 786, 'Jaguar XJ 5.0 L V8 Supercharged': 787, 'Jeep Compass 1.4 Limited Option': 788, 'Jeep Compass 2.0 Limited': 789, 'Jeep Compass 2.0 Limited 4X4': 790, 'Jeep Compass 2.0 Limited Option 4X4': 791, 'Jeep Compass 2.0 Limited Option Black': 792, 'Jeep Compass 2.0 Longitude': 793, 'Jeep Compass 2.0 Sport': 794, 'Lamborghini Gallardo Coupe': 795, 'Land Rover Discovery 3 TDV6 Diesel Automatic': 796, 'Land Rover Discovery 4 TDV6 SE': 797, 'Land Rover Discovery HSE Luxury 3.0 TD6': 798, 'Land Rover Discovery SE 3.0 TD6': 799, 'Land Rover Discovery Sport SD4 HSE Luxury': 800, 'Land Rover Discovery Sport SD4 HSE Luxury 7S': 801, 'Land Rover Discovery Sport TD4 HSE': 802, 'Land Rover Discovery Sport TD4 HSE 7S': 803, 'Land Rover Discovery Sport TD4 S': 804, 'Land Rover Freelander 2 HSE': 805, 'Land Rover Freelander 2 HSE SD4': 806, 'Land Rover Freelander 2 SE': 807, 'Land Rover Freelander 2 TD4 HSE': 808, 'Land Rover Freelander 2 TD4 S': 809, 'Land Rover Freelander 2 TD4 SE': 810, 'Land Rover Range Rover 2.2L Dynamic': 811, 'Land Rover Range Rover 2.2L Prestige': 812, 'Land Rover Range Rover 2.2L Pure': 813, 'Land Rover Range Rover 3.0 D': 814, 'Land Rover Range Rover 3.0 Diesel LWB Vogue': 815, 'Land Rover Range Rover 3.6 TDV8 Vogue SE': 816, 'Land Rover Range Rover 3.6 TDV8 Vogue SE Diesel': 817, 'Land Rover Range Rover Evoque 2.0 TD4 HSE Dynamic': 818, 'Land Rover Range Rover Evoque 2.0 TD4 Pure': 819, 'Land Rover Range Rover HSE Dynamic': 820, 'Land Rover Range Rover Sport 2005 2012 HSE': 821, 'Land Rover Range Rover Sport 2005 2012 Sport': 822, 'Land Rover Range Rover Sport HSE': 823, 'Land Rover Range Rover Sport SE': 824, 'Land Rover Range Rover TDV8 (Diesel)': 825, 'Land Rover Range Rover Vogue SE 4.4 SDV8': 826, 'Mahindra Bolero DI BSII': 827, 'Mahindra Bolero SLE': 828, 'Mahindra Bolero SLE BSIII': 829, 'Mahindra Bolero SLX 2WD': 830, 'Mahindra Bolero VLX BS IV': 831, 'Mahindra Bolero VLX CRDe': 832, 'Mahindra Bolero ZLX': 833, 'Mahindra Bolero ZLX BSIII': 834, 'Mahindra Bolero mHAWK D70 ZLX': 835, 'Mahindra E Verito D4': 836, 'Mahindra Jeep MM 540 DP': 837, 'Mahindra Jeep MM 550 PE': 838, 'Mahindra KUV 100 G80 K6 Plus 5Str': 839, 'Mahindra KUV 100 G80 K8': 840, 'Mahindra KUV 100 G80 K8 Dual Tone': 841, 'Mahindra KUV 100 mFALCON D75 K4': 842, 'Mahindra KUV 100 mFALCON D75 K4 Plus': 843, 'Mahindra KUV 100 mFALCON D75 K4 Plus 5str': 844, 'Mahindra KUV 100 mFALCON D75 K6': 845, 'Mahindra KUV 100 mFALCON D75 K8': 846, 'Mahindra KUV 100 mFALCON D75 K8 Dual Tone': 847, 'Mahindra KUV 100 mFALCON G80 K2': 848, 'Mahindra KUV 100 mFALCON G80 K6 5str AW': 849, 'Mahindra KUV 100 mFALCON G80 K6 Plus': 850, 'Mahindra KUV 100 mFALCON G80 K8': 851, 'Mahindra Logan Diesel 1.5 DLS': 852, 'Mahindra Logan Petrol 1.4 GLE': 853, 'Mahindra NuvoSport N6': 854, 'Mahindra NuvoSport N8': 855, 'Mahindra Quanto C2': 856, 'Mahindra Quanto C4': 857, 'Mahindra Quanto C6': 858, 'Mahindra Quanto C8': 859, 'Mahindra Renault Logan 1.4 GLX Petrol': 860, 'Mahindra Renault Logan 1.5 DLE Diesel': 861, 'Mahindra Scorpio 1.99 S10': 862, 'Mahindra Scorpio 1.99 S10 4WD': 863, 'Mahindra Scorpio 1.99 S4 Plus': 864, 'Mahindra Scorpio 1.99 S8': 865, 'Mahindra Scorpio 2.6 CRDe': 866, 'Mahindra Scorpio 2.6 DX': 867, 'Mahindra Scorpio 2.6 LX': 868, 'Mahindra Scorpio 2.6 SLX CRDe': 869, 'Mahindra Scorpio 2009-2014 LX 2WD 7S': 870, 'Mahindra Scorpio 2009-2014 SLE 7S BSIV': 871, 'Mahindra Scorpio 2009-2014 VLX 2WD 7S BSIV': 872, 'Mahindra Scorpio 2009-2014 VLX 4WD AT 7S BSIV': 873, 'Mahindra Scorpio DX 2.6 Turbo 8 Str': 874, 'Mahindra Scorpio LX 2.6 Turbo': 875, 'Mahindra Scorpio LX BS IV': 876, 'Mahindra Scorpio S10 4WD': 877, 'Mahindra Scorpio S10 7 Seater': 878, 'Mahindra Scorpio S10 AT 4WD': 879, 'Mahindra Scorpio S2 7 Seater': 880, 'Mahindra Scorpio S4 7 Seater': 881, 'Mahindra Scorpio S4 Plus': 882, 'Mahindra Scorpio S6 7 Seater': 883, 'Mahindra Scorpio S6 Plus 7 Seater': 884, 'Mahindra Scorpio S8 8 Seater': 885, 'Mahindra Scorpio SLE BSIII': 886, 'Mahindra Scorpio SLE BSIV': 887, 'Mahindra Scorpio SLX': 888, 'Mahindra Scorpio VLX': 889, 'Mahindra Scorpio VLX 2.2 mHawk Airbag BSIV': 890, 'Mahindra Scorpio VLX 2.2 mHawk BSIII': 891, 'Mahindra Scorpio VLX 2WD AIRBAG BSIV': 892, 'Mahindra Scorpio VLX 2WD AT BSIV': 893, 'Mahindra Scorpio VLX 2WD Airbag BSIII': 894, 'Mahindra Scorpio VLX 2WD BSIV': 895, 'Mahindra Scorpio VLX 4WD': 896, 'Mahindra Scorpio VLX 4WD AIRBAG AT BSIV': 897, 'Mahindra Scorpio VLX 4WD AIRBAG BSIV': 898, 'Mahindra Scorpio VLX AT AIRBAG BSIV': 899, 'Mahindra Ssangyong Rexton RX5': 900, 'Mahindra Ssangyong Rexton RX7': 901, 'Mahindra TUV 300 2015-2019 T8': 902, 'Mahindra TUV 300 2015-2019 mHAWK100 T8': 903, 'Mahindra TUV 300 2015-2019 mHAWK100 T8 AMT': 904, 'Mahindra TUV 300 T8': 905, 'Mahindra TUV 300 T8 AMT': 906, 'Mahindra Thar CRDe': 907, 'Mahindra Thar CRDe AC': 908, 'Mahindra Thar DI 4X4': 909, 'Mahindra Verito 1.5 D4 BSIV': 910, 'Mahindra Verito 1.5 D6 BSIV': 911, 'Mahindra XUV300 W8 Diesel': 912, 'Mahindra XUV500 AT W10 AWD': 913, 'Mahindra XUV500 AT W10 FWD': 914, 'Mahindra XUV500 AT W8 FWD': 915, 'Mahindra XUV500 W10 1.99 mHawk': 916, 'Mahindra XUV500 W10 2WD': 917, 'Mahindra XUV500 W10 AWD': 918, 'Mahindra XUV500 W4': 919, 'Mahindra XUV500 W4 1.99 mHawk': 920, 'Mahindra XUV500 W6 2WD': 921, 'Mahindra XUV500 W7': 922, 'Mahindra XUV500 W8 1.99 mHawk': 923, 'Mahindra XUV500 W8 2WD': 924, 'Mahindra XUV500 W8 4WD': 925, 'Mahindra XUV500 W9 AT': 926, 'Mahindra Xylo D2 BS III': 927, 'Mahindra Xylo D2 BSIV': 928, 'Mahindra Xylo D2 Maxx': 929, 'Mahindra Xylo D4': 930, 'Mahindra Xylo D4 BS III': 931, 'Mahindra Xylo D4 BSIV': 932, 'Mahindra Xylo E2': 933, 'Mahindra Xylo E4': 934, 'Mahindra Xylo E4 ABS BS III': 935, 'Mahindra Xylo E8': 936, 'Mahindra Xylo E8 ABS BS IV': 937, 'Mahindra Xylo E8 BS IV': 938, 'Mahindra Xylo H4': 939, 'Maruti 1000 AC': 940, 'Maruti 800 AC': 941, 'Maruti 800 AC BSIII': 942, 'Maruti 800 AC Uniq': 943, 'Maruti 800 DX BSII': 944, 'Maruti 800 Std': 945, 'Maruti 800 Std BSIII': 946, 'Maruti A-Star AT VXI': 947, 'Maruti A-Star AT Vxi Aktiv': 948, 'Maruti A-Star Lxi': 949, 'Maruti A-Star Vxi': 950, 'Maruti Alto 800 2016-2019 CNG LXI': 951, 'Maruti Alto 800 2016-2019 LXI': 952, 'Maruti Alto 800 2016-2019 LXI Optional': 953, 'Maruti Alto 800 2016-2019 VXI': 954, 'Maruti Alto 800 CNG LXI': 955, 'Maruti Alto 800 LXI': 956, 'Maruti Alto 800 LXI Airbag': 957, 'Maruti Alto 800 VXI': 958, 'Maruti Alto Green LXi (CNG)': 959, 'Maruti Alto K10 2010-2014 VXI': 960, 'Maruti Alto K10 LXI': 961, 'Maruti Alto K10 LXI CNG': 962, 'Maruti Alto K10 LXI CNG Optional': 963, 'Maruti Alto K10 LXI Optional': 964, 'Maruti Alto K10 VXI': 965, 'Maruti Alto K10 VXI AGS': 966, 'Maruti Alto LX': 967, 'Maruti Alto LX BSIII': 968, 'Maruti Alto LXI': 969, 'Maruti Alto LXi': 970, 'Maruti Alto LXi BSII': 971, 'Maruti Alto LXi BSIII': 972, 'Maruti Alto Std': 973, 'Maruti Alto VXi': 974, 'Maruti Alto Vxi 1.1': 975, 'Maruti Baleno Alpha': 976, 'Maruti Baleno Alpha 1.2': 977, 'Maruti Baleno Alpha 1.3': 978, 'Maruti Baleno Alpha Automatic': 979, 'Maruti Baleno Alpha CVT': 980, 'Maruti Baleno Alpha Diesel': 981, 'Maruti Baleno Delta': 982, 'Maruti Baleno Delta 1.2': 983, 'Maruti Baleno Delta 1.3': 984, 'Maruti Baleno Delta Automatic': 985, 'Maruti Baleno Delta CVT': 986, 'Maruti Baleno LXI': 987, 'Maruti Baleno LXI - BSIII': 988, 'Maruti Baleno RS 1.0 Petrol': 989, 'Maruti Baleno RS Petrol': 990, 'Maruti Baleno Sigma 1.2': 991, 'Maruti Baleno Vxi': 992, 'Maruti Baleno Vxi - BSIII': 993, 'Maruti Baleno Zeta': 994, 'Maruti Baleno Zeta 1.2': 995, 'Maruti Baleno Zeta 1.3': 996, 'Maruti Baleno Zeta Automatic': 997, 'Maruti Baleno Zeta CVT': 998, 'Maruti Celerio CNG VXI MT': 999, 'Maruti Celerio LDi': 1000, 'Maruti Celerio LXI': 1001, 'Maruti Celerio LXI MT': 1002, 'Maruti Celerio VXI': 1003, 'Maruti Celerio VXI AMT': 1004, 'Maruti Celerio VXI AT': 1005, 'Maruti Celerio VXI AT Optional': 1006, 'Maruti Celerio VXI Optional AMT': 1007, 'Maruti Celerio ZDi': 1008, 'Maruti Celerio ZXI': 1009, 'Maruti Celerio ZXI AMT': 1010, 'Maruti Celerio ZXI AT': 1011, 'Maruti Celerio ZXI AT Optional': 1012, 'Maruti Celerio ZXI MT': 1013, 'Maruti Celerio ZXI Optional': 1014, 'Maruti Celerio ZXI Optional AMT': 1015, 'Maruti Ciaz 1.3 Alpha': 1016, 'Maruti Ciaz 1.3 S': 1017, 'Maruti Ciaz 1.4 AT Alpha': 1018, 'Maruti Ciaz 1.4 Alpha': 1019, 'Maruti Ciaz 1.4 Delta': 1020, 'Maruti Ciaz 1.4 Zeta': 1021, 'Maruti Ciaz AT ZXi': 1022, 'Maruti Ciaz Alpha': 1023, 'Maruti Ciaz RS ZDi Plus SHVS': 1024, 'Maruti Ciaz RS ZXi Plus': 1025, 'Maruti Ciaz VDI SHVS': 1026, 'Maruti Ciaz VDi': 1027, 'Maruti Ciaz VDi Plus': 1028, 'Maruti Ciaz VDi Plus SHVS': 1029, 'Maruti Ciaz VXi Plus': 1030, 'Maruti Ciaz ZDi': 1031, 'Maruti Ciaz ZDi Plus': 1032, 'Maruti Ciaz ZDi Plus SHVS': 1033, 'Maruti Ciaz ZDi SHVS': 1034, 'Maruti Ciaz ZXi': 1035, 'Maruti Ciaz ZXi Option': 1036, 'Maruti Ciaz ZXi Plus': 1037, 'Maruti Ciaz Zeta': 1038, 'Maruti Dzire AMT VDI': 1039, 'Maruti Dzire AMT VXI': 1040, 'Maruti Dzire AMT ZDI Plus': 1041, 'Maruti Dzire AMT ZXI Plus': 1042, 'Maruti Dzire LDI': 1043, 'Maruti Dzire New': 1044, 'Maruti Dzire VDI': 1045, 'Maruti Dzire VXI': 1046, 'Maruti Dzire ZDI': 1047, 'Maruti Dzire ZDI Plus': 1048, 'Maruti Eeco 5 STR With AC Plus HTR CNG': 1049, 'Maruti Eeco 5 Seater AC': 1050, 'Maruti Eeco 7 Seater Standard': 1051, 'Maruti Eeco CNG 5 Seater AC': 1052, 'Maruti Eeco Smiles 5 Seater AC': 1053, 'Maruti Ertiga LXI': 1054, 'Maruti Ertiga Paseo VDI': 1055, 'Maruti Ertiga Paseo VXI': 1056, 'Maruti Ertiga SHVS VDI': 1057, 'Maruti Ertiga SHVS ZDI Plus': 1058, 'Maruti Ertiga VDI': 1059, 'Maruti Ertiga VDI Limited Edition': 1060, 'Maruti Ertiga VXI': 1061, 'Maruti Ertiga VXI AT Petrol': 1062, 'Maruti Ertiga VXI CNG': 1063, 'Maruti Ertiga ZDI': 1064, 'Maruti Ertiga ZDI Plus': 1065, 'Maruti Ertiga ZXI': 1066, 'Maruti Esteem LX BSII': 1067, 'Maruti Esteem Vxi': 1068, 'Maruti Esteem Vxi - BSIII': 1069, 'Maruti Estilo LXI': 1070, 'Maruti Grand Vitara 2.4 MT': 1071, 'Maruti Grand Vitara AT': 1072, 'Maruti Grand Vitara MT': 1073, 'Maruti Ignis 1.2 AMT Alpha': 1074, 'Maruti Ignis 1.2 Alpha': 1075, 'Maruti Ignis 1.2 Delta': 1076, 'Maruti Ignis 1.3 Zeta': 1077, 'Maruti Omni 5 Seater BSIV': 1078, 'Maruti Omni 5 Str STD': 1079, 'Maruti Omni 8 Seater BSII': 1080, 'Maruti Omni 8 Seater BSIV': 1081, 'Maruti Omni E 8 Str STD': 1082, 'Maruti Omni E MPI STD BS IV': 1083, 'Maruti Omni MPI CARGO BSIII W/ IMMOBILISER': 1084, 'Maruti Omni MPI CARGO BSIV': 1085, 'Maruti Omni MPI STD BSIV': 1086, 'Maruti Ritz AT': 1087, 'Maruti Ritz LDi': 1088, 'Maruti Ritz LXI': 1089, 'Maruti Ritz LXi': 1090, 'Maruti Ritz VDI (ABS) BS IV': 1091, 'Maruti Ritz VDi': 1092, 'Maruti Ritz VXI': 1093, 'Maruti Ritz VXI ABS': 1094, 'Maruti Ritz VXi': 1095, 'Maruti Ritz ZDi': 1096, 'Maruti Ritz ZXI': 1097, 'Maruti Ritz ZXi': 1098, 'Maruti S Cross DDiS 200 Alpha': 1099, 'Maruti S Cross DDiS 200 Delta': 1100, 'Maruti S Cross DDiS 200 Sigma Option': 1101, 'Maruti S Cross DDiS 320 Alpha': 1102, 'Maruti S-Cross Alpha DDiS 200 SH': 1103, 'Maruti S-Cross Delta DDiS 200 SH': 1104, 'Maruti S-Cross Zeta DDiS 200 SH': 1105, 'Maruti SX4 Green Vxi (CNG)': 1106, 'Maruti SX4 S Cross DDiS 200 Zeta': 1107, 'Maruti SX4 S Cross DDiS 320 Zeta': 1108, 'Maruti SX4 VDI': 1109, 'Maruti SX4 Vxi BSIII': 1110, 'Maruti SX4 Vxi BSIV': 1111, 'Maruti SX4 ZDI': 1112, 'Maruti SX4 ZDI Leather': 1113, 'Maruti SX4 ZXI AT Leather': 1114, 'Maruti SX4 ZXI MT BSIV': 1115, 'Maruti SX4 Zxi BSIII': 1116, 'Maruti SX4 Zxi with Leather BSIII': 1117, 'Maruti Swift 1.3 LXI': 1118, 'Maruti Swift 1.3 VXI ABS': 1119, 'Maruti Swift 1.3 VXi': 1120, 'Maruti Swift 1.3 ZXI': 1121, 'Maruti Swift AMT DDiS VDI': 1122, 'Maruti Swift AMT VDI': 1123, 'Maruti Swift DDiS VDI': 1124, 'Maruti Swift DDiS ZDI Plus': 1125, 'Maruti Swift Dzire 1.2 Lxi BSIV': 1126, 'Maruti Swift Dzire 1.2 Vxi BSIV': 1127, 'Maruti Swift Dzire AMT ZDI': 1128, 'Maruti Swift Dzire LDI': 1129, 'Maruti Swift Dzire LDi': 1130, 'Maruti Swift Dzire LXI': 1131, 'Maruti Swift Dzire LXi': 1132, 'Maruti Swift Dzire Ldi BSIV': 1133, 'Maruti Swift Dzire Tour LDI': 1134, 'Maruti Swift Dzire VDI': 1135, 'Maruti Swift Dzire VDI Optional': 1136, 'Maruti Swift Dzire VDi': 1137, 'Maruti Swift Dzire VXI': 1138, 'Maruti Swift Dzire VXI Optional': 1139, 'Maruti Swift Dzire VXi': 1140, 'Maruti Swift Dzire VXi AT': 1141, 'Maruti Swift Dzire Vdi BSIV': 1142, 'Maruti Swift Dzire ZDI': 1143, 'Maruti Swift Dzire ZDi': 1144, 'Maruti Swift Dzire ZXI': 1145, 'Maruti Swift Dzire ZXi': 1146, 'Maruti Swift LDI': 1147, 'Maruti Swift LDI BSIV': 1148, 'Maruti Swift LDI Optional': 1149, 'Maruti Swift LDI SP Limited Edition': 1150, 'Maruti Swift LXI': 1151, 'Maruti Swift LXI BSIV': 1152, 'Maruti Swift LXI Option': 1153, 'Maruti Swift LXi BSIV': 1154, 'Maruti Swift Ldi BSIII': 1155, 'Maruti Swift Ldi BSIV': 1156, 'Maruti Swift Lxi BSIII': 1157, 'Maruti Swift RS VDI': 1158, 'Maruti Swift VDI': 1159, 'Maruti Swift VDI BSIV': 1160, 'Maruti Swift VDI BSIV W ABS': 1161, 'Maruti Swift VDI Optional': 1162, 'Maruti Swift VDi BSIII W/ ABS': 1163, 'Maruti Swift VVT VXI': 1164, 'Maruti Swift VXI': 1165, 'Maruti Swift VXI BSIII': 1166, 'Maruti Swift VXI BSIII W/ ABS': 1167, 'Maruti Swift VXI BSIV': 1168, 'Maruti Swift VXI Optional': 1169, 'Maruti Swift VXi BSIV': 1170, 'Maruti Swift Vdi BSIII': 1171, 'Maruti Swift ZDI': 1172, 'Maruti Swift ZDi': 1173, 'Maruti Swift ZXI': 1174, 'Maruti Swift ZXI ABS': 1175, 'Maruti Swift ZXI Plus': 1176, 'Maruti Versa DX2 BS II': 1177, 'Maruti Vitara Brezza LDi': 1178, 'Maruti Vitara Brezza LDi Option': 1179, 'Maruti Vitara Brezza VDi': 1180, 'Maruti Vitara Brezza VDi Option': 1181, 'Maruti Vitara Brezza ZDi': 1182, 'Maruti Vitara Brezza ZDi Plus': 1183, 'Maruti Vitara Brezza ZDi Plus AMT Dual Tone': 1184, 'Maruti Vitara Brezza ZDi Plus Dual Tone': 1185, 'Maruti Wagon R AMT VXI': 1186, 'Maruti Wagon R AMT VXI Option': 1187, 'Maruti Wagon R CNG LXI': 1188, 'Maruti Wagon R Duo LX BSIII': 1189, 'Maruti Wagon R Duo Lxi': 1190, 'Maruti Wagon R LX': 1191, 'Maruti Wagon R LX BS IV': 1192, 'Maruti Wagon R LX BSII': 1193, 'Maruti Wagon R LX BSIII': 1194, 'Maruti Wagon R LX DUO BSIII': 1195, 'Maruti Wagon R LXI': 1196, 'Maruti Wagon R LXI BS IV': 1197, 'Maruti Wagon R LXI BSII': 1198, 'Maruti Wagon R LXI BSIII': 1199, 'Maruti Wagon R LXI CNG': 1200, 'Maruti Wagon R LXI DUO BSIII': 1201, 'Maruti Wagon R LXI LPG BSIV': 1202, 'Maruti Wagon R LXI Minor': 1203, 'Maruti Wagon R LXI Optional': 1204, 'Maruti Wagon R Stingray LXI': 1205, 'Maruti Wagon R Stingray VXI': 1206, 'Maruti Wagon R VXI': 1207, 'Maruti Wagon R VXI 1.2': 1208, 'Maruti Wagon R VXI AMT': 1209, 'Maruti Wagon R VXI AMT1.2': 1210, 'Maruti Wagon R VXI BS IV': 1211, 'Maruti Wagon R VXI BS IV with ABS': 1212, 'Maruti Wagon R VXI BSIII': 1213, 'Maruti Wagon R VXI Minor': 1214, 'Maruti Wagon R VXI Minor ABS': 1215, 'Maruti Wagon R VXI Plus': 1216, 'Maruti Wagon R VXi BSII': 1217, 'Maruti Wagon R Vx': 1218, 'Maruti Wagon R ZXI AMT 1.2': 1219, 'Maruti Zen Estilo 1.1 LX BSIII': 1220, 'Maruti Zen Estilo 1.1 LXI BSIII': 1221, 'Maruti Zen Estilo 1.1 VXI BSIII': 1222, 'Maruti Zen Estilo LXI BS IV': 1223, 'Maruti Zen Estilo LXI BSIII': 1224, 'Maruti Zen Estilo LXI Green (CNG)': 1225, 'Maruti Zen Estilo VXI BSIII': 1226, 'Maruti Zen LX': 1227, 'Maruti Zen LXI': 1228, 'Maruti Zen LXi - BS III': 1229, 'Maruti Zen LXi BSII': 1230, 'Maruti Zen VX': 1231, 'Maruti Zen VXI': 1232, 'Maruti Zen VXI BSII': 1233, 'Maruti Zen VXi - BS III': 1234, 'Mercedes-Benz A Class A180 CDI': 1235, 'Mercedes-Benz A Class A180 Sport': 1236, 'Mercedes-Benz A Class A200 CDI Sport': 1237, 'Mercedes-Benz B Class 2012-2015 B200 Sport CDI': 1238, 'Mercedes-Benz B Class B180': 1239, 'Mercedes-Benz B Class B200 CDI': 1240, 'Mercedes-Benz C-Class Progressive C 220d': 1241, 'Mercedes-Benz CLA 200 CDI Sport': 1242, 'Mercedes-Benz CLA 200 CDI Style': 1243, 'Mercedes-Benz CLA 200 CGI Sport': 1244, 'Mercedes-Benz CLA 200 D Sport Edition': 1245, 'Mercedes-Benz CLA 200 Sport Edition': 1246, 'Mercedes-Benz CLS-Class 2006-2010 350 CDI': 1247, 'Mercedes-Benz E-Class 200 Kompressor Elegance': 1248, 'Mercedes-Benz E-Class 2009-2013 E 200 CGI Avantgarde': 1249, 'Mercedes-Benz E-Class 2009-2013 E 200 CGI Elegance': 1250, 'Mercedes-Benz E-Class 2009-2013 E 220 CDI Avantgarde': 1251, 'Mercedes-Benz E-Class 2009-2013 E 250 Elegance': 1252, 'Mercedes-Benz E-Class 2009-2013 E200 CGI Blue Efficiency': 1253, 'Mercedes-Benz E-Class 2009-2013 E250 CDI Avantgarde': 1254, 'Mercedes-Benz E-Class 2009-2013 E250 CDI Blue Efficiency': 1255, 'Mercedes-Benz E-Class 2009-2013 E250 CDI Classic': 1256, 'Mercedes-Benz E-Class 2009-2013 E250 CDI Elegance': 1257, 'Mercedes-Benz E-Class 2009-2013 E350 CDI Avantgarde': 1258, 'Mercedes-Benz E-Class 2009-2013 E350 Petrol': 1259, 'Mercedes-Benz E-Class 2015-2017 E 200 CGI': 1260, 'Mercedes-Benz E-Class 2015-2017 E250 CDI Avantgarde': 1261, 'Mercedes-Benz E-Class 2015-2017 E250 Edition E': 1262, 'Mercedes-Benz E-Class 2015-2017 E350 CDI Avantgrade': 1263, 'Mercedes-Benz E-Class 2015-2017 E350 Edition E': 1264, 'Mercedes-Benz E-Class 220 CDI': 1265, 'Mercedes-Benz E-Class 230': 1266, 'Mercedes-Benz E-Class 230 E AT': 1267, 'Mercedes-Benz E-Class 250 D W 210': 1268, 'Mercedes-Benz E-Class 280 CDI': 1269, 'Mercedes-Benz E-Class 280 Elegance': 1270, 'Mercedes-Benz E-Class E 200 CGI': 1271, 'Mercedes-Benz E-Class E 220 d': 1272, 'Mercedes-Benz E-Class E 350 d': 1273, 'Mercedes-Benz E-Class E250 CDI Avantgrade': 1274, 'Mercedes-Benz E-Class E250 CDI Launch Edition': 1275, 'Mercedes-Benz E-Class E270 CDI': 1276, 'Mercedes-Benz E-Class E350 CDI': 1277, 'Mercedes-Benz E-Class E400 Cabriolet': 1278, 'Mercedes-Benz E-Class Facelift': 1279, 'Mercedes-Benz GL-Class 2007 2012 350 CDI Luxury': 1280, 'Mercedes-Benz GL-Class 350 CDI Blue Efficiency': 1281, 'Mercedes-Benz GLA Class 200 CDI': 1282, 'Mercedes-Benz GLA Class 200 CDI SPORT': 1283, 'Mercedes-Benz GLA Class 200 Sport': 1284, 'Mercedes-Benz GLA Class 200 Sport Edition': 1285, 'Mercedes-Benz GLA Class 200 d Sport': 1286, 'Mercedes-Benz GLA Class 200 d Style': 1287, 'Mercedes-Benz GLC 220': 1288, 'Mercedes-Benz GLC 220d 4MATIC Sport': 1289, 'Mercedes-Benz GLC 220d 4MATIC Style': 1290, 'Mercedes-Benz GLC 220d Celebration Edition': 1291, 'Mercedes-Benz GLC 43 AMG Coupe': 1292, 'Mercedes-Benz GLE 250d': 1293, 'Mercedes-Benz GLE 350d': 1294, 'Mercedes-Benz GLS 350d 4MATIC': 1295, 'Mercedes-Benz GLS 350d Grand Edition': 1296, 'Mercedes-Benz M-Class ML 250 CDI': 1297, 'Mercedes-Benz M-Class ML 320 CDI': 1298, 'Mercedes-Benz M-Class ML 350 4Matic': 1299, 'Mercedes-Benz M-Class ML 350 CDI': 1300, 'Mercedes-Benz M-Class ML 350 CDI Corporate': 1301, 'Mercedes-Benz New C-Class 200 CDI Classic': 1302, 'Mercedes-Benz New C-Class 200 CDI Elegance': 1303, 'Mercedes-Benz New C-Class 200 K AT': 1304, 'Mercedes-Benz New C-Class 200 Kompressor': 1305, 'Mercedes-Benz New C-Class 220': 1306, 'Mercedes-Benz New C-Class 220 CDI AT': 1307, 'Mercedes-Benz New C-Class 220 CDI MT': 1308, 'Mercedes-Benz New C-Class 230 Avantgarde': 1309, 'Mercedes-Benz New C-Class 250 CDI Classic': 1310, 'Mercedes-Benz New C-Class C 200 AVANTGARDE': 1311, 'Mercedes-Benz New C-Class C 200 Avantgarde Edition C': 1312, 'Mercedes-Benz New C-Class C 200 CGI': 1313, 'Mercedes-Benz New C-Class C 200 CGI Avantgarde': 1314, 'Mercedes-Benz New C-Class C 200 CGI Elegance': 1315, 'Mercedes-Benz New C-Class C 200 Kompressor Elegance AT': 1316, 'Mercedes-Benz New C-Class C 200 Kompressor Elegance MT': 1317, 'Mercedes-Benz New C-Class C 220 CDI Avantgarde': 1318, 'Mercedes-Benz New C-Class C 220 CDI BE Avantgare': 1319, 'Mercedes-Benz New C-Class C 220 CDI CLASSIC': 1320, 'Mercedes-Benz New C-Class C 220 CDI Celebration Edition': 1321, 'Mercedes-Benz New C-Class C 220 CDI Elegance AT': 1322, 'Mercedes-Benz New C-Class C 220 CDI Style': 1323, 'Mercedes-Benz New C-Class C 220CDIBE Avantgarde Command': 1324, 'Mercedes-Benz New C-Class C 220d Avantgarde Edition C': 1325, 'Mercedes-Benz New C-Class C 250 CDI Avantgarde': 1326, 'Mercedes-Benz New C-Class C 250 CDI Elegance': 1327, 'Mercedes-Benz New C-Class C220 CDI Executive Edition': 1328, 'Mercedes-Benz New C-Class C250 Avantgarde': 1329, 'Mercedes-Benz New C-Class Progressive C 200': 1330, 'Mercedes-Benz R-Class R350 4Matic Long': 1331, 'Mercedes-Benz R-Class R350 CDI 4MATIC': 1332, 'Mercedes-Benz S Class 2005 2013 320 CDI': 1333, 'Mercedes-Benz S Class 2005 2013 S 300 L': 1334, 'Mercedes-Benz S Class 2005 2013 S 350 CDI': 1335, 'Mercedes-Benz S Class 2005 2013 S 350 L': 1336, 'Mercedes-Benz S Class 2005 2013 S 500': 1337, 'Mercedes-Benz S-Class 280 AT': 1338, 'Mercedes-Benz S-Class 320 L': 1339, 'Mercedes-Benz S-Class S 350 d': 1340, 'Mercedes-Benz SL-Class SL 500': 1341, 'Mercedes-Benz SLC 43 AMG': 1342, 'Mercedes-Benz SLK-Class 55 AMG': 1343, 'Mercedes-Benz SLK-Class SLK 350': 1344, 'Mini Clubman Cooper S': 1345, 'Mini Cooper 3 DOOR D': 1346, 'Mini Cooper 3 DOOR S': 1347, 'Mini Cooper 5 DOOR D': 1348, 'Mini Cooper Convertible 1.6': 1349, 'Mini Cooper Convertible S': 1350, 'Mini Cooper Countryman D': 1351, 'Mini Cooper Countryman D High': 1352, 'Mini Cooper S': 1353, 'Mini Cooper S Carbon Edition': 1354, 'Mini Countryman Cooper D': 1355, 'Mitsubishi Cedia Sports': 1356, 'Mitsubishi Lancer 1.5 SFXi': 1357, 'Mitsubishi Lancer GLXD': 1358, 'Mitsubishi Montero 3.2 GLS': 1359, 'Mitsubishi Outlander 2.4': 1360, 'Mitsubishi Pajero 2.8 GLX CRZ': 1361, 'Mitsubishi Pajero 2.8 GLX Sports': 1362, 'Mitsubishi Pajero 2.8 SFX': 1363, 'Mitsubishi Pajero 2.8 SFX 7Str': 1364, 'Mitsubishi Pajero 2.8 SFX BSIII Dual Tone': 1365, 'Mitsubishi Pajero 2.8 SFX BSIV Dual Tone': 1366, 'Mitsubishi Pajero 4X4 LHD': 1367, 'Mitsubishi Pajero Sport 4X2 AT DualTone BlackTop': 1368, 'Mitsubishi Pajero Sport 4X4': 1369, 'Mitsubishi Pajero Sport 4X4 AT': 1370, 'Mitsubishi Pajero Sport Anniversary Edition': 1371, 'Nissan Evalia 2013 XL': 1372, 'Nissan Micra Active XL Petrol': 1373, 'Nissan Micra Active XV': 1374, 'Nissan Micra Active XV S': 1375, 'Nissan Micra Diesel': 1376, 'Nissan Micra Diesel XL': 1377, 'Nissan Micra Diesel XV': 1378, 'Nissan Micra Diesel XV Premium': 1379, 'Nissan Micra Diesel XV Primo': 1380, 'Nissan Micra XE': 1381, 'Nissan Micra XL': 1382, 'Nissan Micra XL Primo': 1383, 'Nissan Micra XV': 1384, 'Nissan Micra XV CVT': 1385, 'Nissan Micra XV D': 1386, 'Nissan Micra XV Primo': 1387, 'Nissan Sunny 2011-2014 Diesel XL': 1388, 'Nissan Sunny 2011-2014 Diesel XV': 1389, 'Nissan Sunny 2011-2014 XL': 1390, 'Nissan Sunny 2011-2014 XL AT Special Edition': 1391, 'Nissan Sunny 2011-2014 XV': 1392, 'Nissan Sunny 2011-2014 XV Special Edition': 1393, 'Nissan Sunny Diesel XV': 1394, 'Nissan Sunny XE P': 1395, 'Nissan Sunny XL': 1396, 'Nissan Sunny XL CVT': 1397, 'Nissan Sunny XV CVT': 1398, 'Nissan Sunny XV D': 1399, 'Nissan Teana 230jM': 1400, 'Nissan Teana XV': 1401, 'Nissan Terrano XL': 1402, 'Nissan Terrano XL 110 PS': 1403, 'Nissan Terrano XL 85 PS': 1404, 'Nissan Terrano XL D Option': 1405, 'Nissan Terrano XL Plus 85 PS': 1406, 'Nissan Terrano XL Plus ICC WT20 SE': 1407, 'Nissan Terrano XV 110 PS': 1408, 'Nissan Terrano XV 110 PS Limited Edition': 1409, 'Nissan Terrano XV D Pre': 1410, 'Nissan Terrano XV Premium 110 PS': 1411, 'Nissan Terrano XV Premium 110 PS Anniversary Edition': 1412, 'Nissan X-Trail SLX AT': 1413, 'Nissan X-Trail SLX MT': 1414, 'Porsche Boxster S tiptronic': 1415, 'Porsche Cayenne 2009-2014 Diesel': 1416, 'Porsche Cayenne 2009-2014 Turbo': 1417, 'Porsche Cayenne Base': 1418, 'Porsche Cayenne Diesel': 1419, 'Porsche Cayenne S Diesel': 1420, 'Porsche Cayenne Turbo S': 1421, 'Porsche Cayman 2009-2012 S': 1422, 'Porsche Cayman 2009-2012 S tiptronic': 1423, 'Porsche Panamera 2010 2013 4S': 1424, 'Porsche Panamera 2010 2013 Diesel': 1425, 'Porsche Panamera Diesel': 1426, 'Porsche Panamera Diesel 250hp': 1427, 'Renault Captur 1.5 Diesel Platine Mono': 1428, 'Renault Captur 1.5 Diesel RXL': 1429, 'Renault Duster 110PS Diesel RXZ Option': 1430, 'Renault Duster 110PS Diesel RXZ Optional with Nav': 1431, 'Renault Duster 110PS Diesel RxL': 1432, 'Renault Duster 110PS Diesel RxL AMT': 1433, 'Renault Duster 110PS Diesel RxZ': 1434, 'Renault Duster 110PS Diesel RxZ AMT': 1435, 'Renault Duster 110PS Diesel RxZ AWD': 1436, 'Renault Duster 110PS Diesel RxZ Pack': 1437, 'Renault Duster 110PS Diesel RxZ Plus': 1438, 'Renault Duster 85PS Diesel RxE': 1439, 'Renault Duster 85PS Diesel RxL': 1440, 'Renault Duster 85PS Diesel RxL Explore': 1441, 'Renault Duster 85PS Diesel RxL Option': 1442, 'Renault Duster 85PS Diesel RxL Optional': 1443, 'Renault Duster 85PS Diesel RxL Optional with Nav': 1444, 'Renault Duster 85PS Diesel RxL Plus': 1445, 'Renault Duster Adventure Edition': 1446, 'Renault Duster Petrol RxL': 1447, 'Renault Duster RXZ AWD': 1448, 'Renault Fluence 1.5': 1449, 'Renault Fluence 2.0': 1450, 'Renault Fluence Diesel E4': 1451, 'Renault KWID 1.0 RXL': 1452, 'Renault KWID 1.0 RXT Optional': 1453, 'Renault KWID 1.0 RXT Optional AMT': 1454, 'Renault KWID AMT RXL': 1455, 'Renault KWID Climber 1.0 AMT': 1456, 'Renault KWID Climber 1.0 MT': 1457, 'Renault KWID RXL': 1458, 'Renault KWID RXT': 1459, 'Renault KWID RXT Optional': 1460, 'Renault Koleos 2.0 Diesel': 1461, 'Renault Lodgy 110PS RxZ 8 Seater': 1462, 'Renault Pulse Petrol RxL': 1463, 'Renault Pulse Petrol RxZ': 1464, 'Renault Pulse RxL': 1465, 'Renault Scala Diesel RxL': 1466, 'Renault Scala RxL': 1467, 'Renault Scala RxL AT': 1468, 'Skoda Fabia 1.2 MPI Ambiente Petrol': 1469, 'Skoda Fabia 1.2 MPI Ambition Plus': 1470, 'Skoda Fabia 1.2 MPI Classic': 1471, 'Skoda Fabia 1.2 Petrol Active': 1472, 'Skoda Fabia 1.2 TDI Ambition': 1473, 'Skoda Fabia 1.2L Diesel Classic': 1474, 'Skoda Fabia 1.4 MPI Ambiente': 1475, 'Skoda Fabia 1.4 TDI Active': 1476, 'Skoda Fabia 1.6 MPI Elegance': 1477, 'Skoda Laura 1.8 TSI Ambiente': 1478, 'Skoda Laura 1.9 TDI AT Ambiente': 1479, 'Skoda Laura 1.9 TDI AT Elegance': 1480, 'Skoda Laura 1.9 TDI MT Ambiente': 1481, 'Skoda Laura Ambiente': 1482, 'Skoda Laura Ambiente 2.0 TDI CR AT': 1483, 'Skoda Laura Ambiente 2.0 TDI CR MT': 1484, 'Skoda Laura Ambition 2.0 TDI CR AT': 1485, 'Skoda Laura Ambition 2.0 TDI CR MT': 1486, 'Skoda Laura Classic 1.8 TSI': 1487, 'Skoda Laura Elegance 2.0 TDI CR AT': 1488, 'Skoda Laura Elegance MT': 1489, 'Skoda Laura L and K AT': 1490, 'Skoda Laura L n K 1.9 PD AT': 1491, 'Skoda Laura RS': 1492, 'Skoda Octavia 1.9 TDI': 1493, 'Skoda Octavia 2.0 TDI AT Style Plus': 1494, 'Skoda Octavia Ambiente 1.9 TDI': 1495, 'Skoda Octavia Ambiente 1.9 TDI MT': 1496, 'Skoda Octavia Ambition 1.4 TSI MT': 1497, 'Skoda Octavia Ambition 1.8 TSI AT': 1498, 'Skoda Octavia Ambition 2.0 TDI AT': 1499, 'Skoda Octavia Ambition Plus 2.0 TDI AT': 1500, 'Skoda Octavia Classic 1.9 TDI MT': 1501, 'Skoda Octavia Elegance 1.8 TSI AT': 1502, 'Skoda Octavia Elegance 1.9 TDI': 1503, 'Skoda Octavia Elegance 2.0 TDI AT': 1504, 'Skoda Octavia L and K 1.9 TDI MT': 1505, 'Skoda Octavia RS': 1506, 'Skoda Octavia Rider 1.9 TDI MT': 1507, 'Skoda Octavia Rider Classic 1.9 TDI': 1508, 'Skoda Octavia Style Plus 2.0 TDI AT': 1509, 'Skoda Rapid 1.5 TDI AT Ambition': 1510, 'Skoda Rapid 1.5 TDI AT Elegance': 1511, 'Skoda Rapid 1.5 TDI AT Elegance Plus': 1512, 'Skoda Rapid 1.5 TDI AT Elegance Plus Black Package': 1513, 'Skoda Rapid 1.5 TDI AT Style': 1514, 'Skoda Rapid 1.5 TDI Ambition': 1515, 'Skoda Rapid 1.5 TDI Ambition Plus': 1516, 'Skoda Rapid 1.5 TDI Elegance': 1517, 'Skoda Rapid 1.5 TDI Style': 1518, 'Skoda Rapid 1.6 MPI AT Ambition': 1519, 'Skoda Rapid 1.6 MPI AT Ambition Plus': 1520, 'Skoda Rapid 1.6 MPI AT Ambition Plus Alloy': 1521, 'Skoda Rapid 1.6 MPI AT Elegance': 1522, 'Skoda Rapid 1.6 MPI AT Elegance Plus': 1523, 'Skoda Rapid 1.6 MPI AT Style': 1524, 'Skoda Rapid 1.6 MPI AT Style Black Package': 1525, 'Skoda Rapid 1.6 MPI AT Style Plus': 1526, 'Skoda Rapid 1.6 MPI Active': 1527, 'Skoda Rapid 1.6 MPI Ambition': 1528, 'Skoda Rapid 1.6 MPI Ambition Plus': 1529, 'Skoda Rapid 1.6 MPI Ambition With Alloy Wheel': 1530, 'Skoda Rapid 1.6 MPI Elegance': 1531, 'Skoda Rapid 1.6 MPI Elegance Black Package': 1532, 'Skoda Rapid 1.6 TDI Active': 1533, 'Skoda Rapid 1.6 TDI Ambition': 1534, 'Skoda Rapid 1.6 TDI Elegance': 1535, 'Skoda Rapid 2013-2016 1.6 MPI Ambition': 1536, 'Skoda Rapid Leisure 1.6 TDI MT': 1537, 'Skoda Rapid Ultima 1.6 MPI Ambition Plus': 1538, 'Skoda Rapid Ultima 1.6 TDI Elegance': 1539, 'Skoda Superb 1.8 TFSI MT': 1540, 'Skoda Superb 1.8 TSI': 1541, 'Skoda Superb 1.8 TSI MT': 1542, 'Skoda Superb 2.5 TDi AT': 1543, 'Skoda Superb 2.8 V6 AT': 1544, 'Skoda Superb 2009-2014 Elegance 2.0 TDI MT': 1545, 'Skoda Superb 3.6 V6 FSI': 1546, 'Skoda Superb Ambition 2.0 TDI CR AT': 1547, 'Skoda Superb Elegance 1.8 TSI AT': 1548, 'Skoda Superb Elegance 1.8 TSI MT': 1549, 'Skoda Superb Elegance 2.0 TDI CR AT': 1550, 'Skoda Superb L&K 1.8 TSI AT': 1551, 'Skoda Superb L&K 2.0 TDI AT': 1552, 'Skoda Superb Style 1.8 TSI AT': 1553, 'Skoda Superb Style 2.0 TDI AT': 1554, 'Skoda Yeti Ambition 4WD': 1555, 'Skoda Yeti Ambition 4X2': 1556, 'Skoda Yeti Elegance': 1557, 'Skoda Yeti Elegance 4X2': 1558, 'Smart Fortwo CDI AT': 1559, 'Tata Bolt Quadrajet XE': 1560, 'Tata Bolt Quadrajet XM': 1561, 'Tata Bolt Revotron XT': 1562, 'Tata Hexa XT': 1563, 'Tata Hexa XTA': 1564, 'Tata Indica DLS': 1565, 'Tata Indica GLS BS IV': 1566, 'Tata Indica LEI': 1567, 'Tata Indica V2 2001-2011 eLS': 1568, 'Tata Indica V2 DL BSIII': 1569, 'Tata Indica V2 DLE BSIII': 1570, 'Tata Indica V2 DLG TC': 1571, 'Tata Indica V2 DLS': 1572, 'Tata Indica V2 DLS BSII': 1573, 'Tata Indica V2 DLS BSIII': 1574, 'Tata Indica V2 DLS TC': 1575, 'Tata Indica V2 DLX': 1576, 'Tata Indica V2 eLS': 1577, 'Tata Indica V2 eLX': 1578, 'Tata Indica Vista Aqua 1.3 Quadrajet': 1579, 'Tata Indica Vista Aura 1.3 Quadrajet': 1580, 'Tata Indica Vista Quadrajet 90 ZX Plus': 1581, 'Tata Indica Vista Quadrajet LS': 1582, 'Tata Indica Vista Quadrajet VX': 1583, 'Tata Indica Vista TDI LS': 1584, 'Tata Indica Vista TDI LX': 1585, 'Tata Indigo CS Emax CNG GLX': 1586, 'Tata Indigo CS LX (TDI) BS III': 1587, 'Tata Indigo CS eGLX BS IV': 1588, 'Tata Indigo CS eGVX': 1589, 'Tata Indigo CS eVX': 1590, 'Tata Indigo GLE': 1591, 'Tata Indigo LS': 1592, 'Tata Indigo LS Dicor': 1593, 'Tata Indigo LX': 1594, 'Tata Indigo LX BSII': 1595, 'Tata Indigo XL Classic Dicor': 1596, 'Tata Indigo XL Grand Petrol': 1597, 'Tata Indigo eCS GLE BSIII': 1598, 'Tata Indigo eCS GLX': 1599, 'Tata Indigo eCS LS (TDI) BS-III': 1600, 'Tata Indigo eCS LX BSIV': 1601, 'Tata Indigo eCS eLS BS IV': 1602, 'Tata Indigo eCS eLX BS IV': 1603, 'Tata Manza Aqua Quadrajet': 1604, 'Tata Manza Aqua Quadrajet BS IV': 1605, 'Tata Manza Aqua Safire': 1606, 'Tata Manza Aura Plus Quadrajet BS IV': 1607, 'Tata Manza Aura Quadrajet': 1608, 'Tata Manza Aura Quadrajet BS IV': 1609, 'Tata Manza Aura Safire': 1610, 'Tata Manza Aura Safire BS IV': 1611, 'Tata Manza Club Class Quadrajet90 LS': 1612, 'Tata Manza Club Class Quadrajet90 LX': 1613, 'Tata Manza ELAN Quadrajet BS IV': 1614, 'Tata Nano CX': 1615, 'Tata Nano Cx': 1616, 'Tata Nano Cx BSIV': 1617, 'Tata Nano LX': 1618, 'Tata Nano LX SE': 1619, 'Tata Nano Lx': 1620, 'Tata Nano Lx BSIV': 1621, 'Tata Nano STD SE': 1622, 'Tata Nano Twist XT': 1623, 'Tata Nano XT': 1624, 'Tata Nano XTA': 1625, 'Tata New Safari 4X4 EX': 1626, 'Tata New Safari DICOR 2.2 EX 4x2': 1627, 'Tata New Safari DICOR 2.2 EX 4x4': 1628, 'Tata New Safari DICOR 2.2 EX 4x4 BS IV': 1629, 'Tata New Safari DICOR 2.2 VX 4x2': 1630, 'Tata New Safari DICOR 2.2 VX 4x4': 1631, 'Tata New Safari Dicor EX 4X2': 1632, 'Tata New Safari EX 4x2': 1633, 'Tata Nexon 1.2 Revotron XZ Plus': 1634, 'Tata Nexon 1.5 Revotorq XZ Plus Dual Tone': 1635, 'Tata Safari DICOR 2.2 LX 4x2': 1636, 'Tata Safari Storme 2012-2015 EX': 1637, 'Tata Safari Storme 2012-2015 LX': 1638, 'Tata Safari Storme 2012-2015 VX': 1639, 'Tata Safari Storme Explorer Edition': 1640, 'Tata Safari Storme VX': 1641, 'Tata Safari Storme VX Varicor 400': 1642, 'Tata Sumo DX': 1643, 'Tata Sumo Delux': 1644, 'Tata Sumo EX': 1645, 'Tata Tiago 1.2 Revotron XM Option': 1646, 'Tata Tiago 1.2 Revotron XT': 1647, 'Tata Tiago 1.2 Revotron XT Option': 1648, 'Tata Tiago 1.2 Revotron XZ': 1649, 'Tata Tiago 1.2 Revotron XZ WO Alloy': 1650, 'Tata Tiago AMT 1.2 Revotron XZA': 1651, 'Tata Tiago Wizz 1.05 Revotorq': 1652, 'Tata Tigor 1.05 Revotorq XT': 1653, 'Tata Tigor 1.05 Revotorq XZ Option': 1654, 'Tata Tigor 1.2 Revotron XT': 1655, 'Tata Tigor 1.2 Revotron XTA': 1656, 'Tata Tigor XE Diesel': 1657, 'Tata Venture EX': 1658, 'Tata Xenon XT EX 4X2': 1659, 'Tata Xenon XT EX 4X4': 1660, 'Tata Zest Quadrajet 1.3': 1661, 'Tata Zest Quadrajet 1.3 75PS XE': 1662, 'Tata Zest Quadrajet 1.3 75PS XM': 1663, 'Tata Zest Quadrajet 1.3 XT': 1664, 'Tata Zest Revotron 1.2 XT': 1665, 'Tata Zest Revotron 1.2T XE': 1666, 'Tata Zest Revotron 1.2T XM': 1667, 'Tata Zest Revotron 1.2T XMS': 1668, 'Toyota Camry 2.5 G': 1669, 'Toyota Camry 2.5 Hybrid': 1670, 'Toyota Camry A/T': 1671, 'Toyota Camry Hybrid': 1672, 'Toyota Camry Hybrid 2.5': 1673, 'Toyota Camry W2 (AT)': 1674, 'Toyota Camry W4 (AT)': 1675, 'Toyota Corolla 1.8 J': 1676, 'Toyota Corolla Altis 1.4 DG': 1677, 'Toyota Corolla Altis 1.8 G': 1678, 'Toyota Corolla Altis 1.8 G CNG': 1679, 'Toyota Corolla Altis 1.8 G CVT': 1680, 'Toyota Corolla Altis 1.8 GL': 1681, 'Toyota Corolla Altis 1.8 Limited Edition': 1682, 'Toyota Corolla Altis 1.8 Sport': 1683, 'Toyota Corolla Altis 1.8 VL CVT': 1684, 'Toyota Corolla Altis 2008-2013 1.8 VL AT': 1685, 'Toyota Corolla Altis Aero D 4D J': 1686, 'Toyota Corolla Altis D-4D G': 1687, 'Toyota Corolla Altis D-4D GL': 1688, 'Toyota Corolla Altis D-4D J': 1689, 'Toyota Corolla Altis Diesel D4DG': 1690, 'Toyota Corolla Altis Diesel D4DJ': 1691, 'Toyota Corolla Altis G': 1692, 'Toyota Corolla Altis G AT': 1693, 'Toyota Corolla Altis G HV AT': 1694, 'Toyota Corolla Altis G MT': 1695, 'Toyota Corolla Altis GL MT': 1696, 'Toyota Corolla Altis JS MT': 1697, 'Toyota Corolla Altis VL': 1698, 'Toyota Corolla Altis VL AT': 1699, 'Toyota Corolla DX': 1700, 'Toyota Corolla Executive (HE)': 1701, 'Toyota Corolla H2': 1702, 'Toyota Corolla H4': 1703, 'Toyota Corolla H5': 1704, 'Toyota Etios 1.4 VXD': 1705, 'Toyota Etios Cross 1.4 GD': 1706, 'Toyota Etios Cross 1.4L GD': 1707, 'Toyota Etios Cross 1.4L VD': 1708, 'Toyota Etios G': 1709, 'Toyota Etios G SP': 1710, 'Toyota Etios G Safety': 1711, 'Toyota Etios GD': 1712, 'Toyota Etios GD SP': 1713, 'Toyota Etios Liva 1.2 G': 1714, 'Toyota Etios Liva 1.2 VX Dual Tone': 1715, 'Toyota Etios Liva 1.4 GD': 1716, 'Toyota Etios Liva Diesel': 1717, 'Toyota Etios Liva G': 1718, 'Toyota Etios Liva GD': 1719, 'Toyota Etios Liva GD SP': 1720, 'Toyota Etios Liva V': 1721, 'Toyota Etios Petrol TRD Sportivo': 1722, 'Toyota Etios V': 1723, 'Toyota Etios VD': 1724, 'Toyota Etios VD SP': 1725, 'Toyota Etios VX': 1726, 'Toyota Etios VXD': 1727, 'Toyota Fortuner 2.8 2WD AT': 1728, 'Toyota Fortuner 2.8 2WD MT': 1729, 'Toyota Fortuner 2.8 4WD MT': 1730, 'Toyota Fortuner 3.0 Diesel': 1731, 'Toyota Fortuner 4x2 4 Speed AT': 1732, 'Toyota Fortuner 4x2 AT': 1733, 'Toyota Fortuner 4x2 AT TRD Sportivo': 1734, 'Toyota Fortuner 4x2 MT TRD Sportivo': 1735, 'Toyota Fortuner 4x2 Manual': 1736, 'Toyota Fortuner 4x4 AT': 1737, 'Toyota Fortuner 4x4 MT': 1738, 'Toyota Fortuner 4x4 MT TRD Sportivo': 1739, 'Toyota Fortuner TRD Sportivo 2.8 2WD AT': 1740, 'Toyota Innova 2.0 E': 1741, 'Toyota Innova 2.0 G1': 1742, 'Toyota Innova 2.0 GX 8 STR': 1743, 'Toyota Innova 2.5 EV Diesel MS 8 Str BSIII': 1744, 'Toyota Innova 2.5 G (Diesel) 7 Seater': 1745, 'Toyota Innova 2.5 G (Diesel) 7 Seater BS IV': 1746, 'Toyota Innova 2.5 G (Diesel) 8 Seater': 1747, 'Toyota Innova 2.5 G (Diesel) 8 Seater BS IV': 1748, 'Toyota Innova 2.5 G3': 1749, 'Toyota Innova 2.5 G4 Diesel 7-seater': 1750, 'Toyota Innova 2.5 G4 Diesel 8-seater': 1751, 'Toyota Innova 2.5 GX (Diesel) 7 Seater': 1752, 'Toyota Innova 2.5 GX (Diesel) 7 Seater BS IV': 1753, 'Toyota Innova 2.5 GX (Diesel) 8 Seater': 1754, 'Toyota Innova 2.5 GX (Diesel) 8 Seater BS IV': 1755, 'Toyota Innova 2.5 GX 7 STR': 1756, 'Toyota Innova 2.5 GX 7 STR BSIV': 1757, 'Toyota Innova 2.5 LE 2014 Diesel 7 Seater': 1758, 'Toyota Innova 2.5 V Diesel 7-seater': 1759, 'Toyota Innova 2.5 V Diesel 8-seater': 1760, 'Toyota Innova 2.5 VX (Diesel) 7 Seater': 1761, 'Toyota Innova 2.5 VX (Diesel) 7 Seater BS IV': 1762, 'Toyota Innova 2.5 VX (Diesel) 8 Seater': 1763, 'Toyota Innova 2.5 VX (Diesel) 8 Seater BS IV': 1764, 'Toyota Innova 2.5 VX 7 STR BSIV': 1765, 'Toyota Innova 2.5 Z Diesel 7 Seater': 1766, 'Toyota Innova 2.5 ZX Diesel 7 Seater': 1767, 'Toyota Innova 2.5 ZX Diesel 7 Seater BSIII': 1768, 'Toyota Innova Crysta 2.4 G MT': 1769, 'Toyota Innova Crysta 2.4 G MT 8S': 1770, 'Toyota Innova Crysta 2.4 GX MT': 1771, 'Toyota Innova Crysta 2.4 GX MT 8S': 1772, 'Toyota Innova Crysta 2.4 VX MT': 1773, 'Toyota Innova Crysta 2.4 VX MT 8S': 1774, 'Toyota Innova Crysta 2.4 ZX MT': 1775, 'Toyota Innova Crysta 2.7 GX MT': 1776, 'Toyota Innova Crysta 2.8 GX AT': 1777, 'Toyota Innova Crysta 2.8 GX AT 8S': 1778, 'Toyota Innova Crysta 2.8 ZX AT': 1779, 'Toyota Platinum Etios 1.4 GXD': 1780, 'Toyota Prius 2009-2016 Z4': 1781, 'Toyota Qualis FS B2': 1782, 'Toyota Qualis FS B3': 1783, 'Toyota Qualis Fleet A3': 1784, 'Toyota Qualis RS E2': 1785, 'Volkswagen Ameo 1.2 MPI Anniversary Edition': 1786, 'Volkswagen Ameo 1.2 MPI Comfortline': 1787, 'Volkswagen Ameo 1.2 MPI Highline': 1788, 'Volkswagen Ameo 1.2 MPI Highline 16 Alloy': 1789, 'Volkswagen Ameo 1.2 MPI Highline Plus 16': 1790, 'Volkswagen Ameo 1.2 MPI Trendline': 1791, 'Volkswagen Ameo 1.5 TDI Comfortline': 1792, 'Volkswagen Ameo 1.5 TDI Comfortline AT': 1793, 'Volkswagen Ameo 1.5 TDI Highline': 1794, 'Volkswagen Ameo 1.5 TDI Highline AT': 1795, 'Volkswagen Beetle 2.0': 1796, 'Volkswagen CrossPolo 1.5 TDI': 1797, 'Volkswagen Jetta 2007-2011 1.9 Highline TDI': 1798, 'Volkswagen Jetta 2007-2011 1.9 L TDI': 1799, 'Volkswagen Jetta 2007-2011 1.9 TDI Trendline': 1800, 'Volkswagen Jetta 2007-2011 2.0 TDI Comfortline': 1801, 'Volkswagen Jetta 2007-2011 2.0 TDI Trendline': 1802, 'Volkswagen Jetta 2012-2014 1.4 TSI': 1803, 'Volkswagen Jetta 2012-2014 2.0L TDI Comfortline': 1804, 'Volkswagen Jetta 2012-2014 2.0L TDI Highline': 1805, 'Volkswagen Jetta 2012-2014 2.0L TDI Highline AT': 1806, 'Volkswagen Jetta 2012-2014 2.0L TDI Trendline': 1807, 'Volkswagen Jetta 2013-2015 2.0L TDI Highline': 1808, 'Volkswagen Jetta 2013-2015 2.0L TDI Highline AT': 1809, 'Volkswagen Passat 1.8 TSI MT': 1810, 'Volkswagen Passat 2.0 PD DSG': 1811, 'Volkswagen Passat Diesel Highline 2.0 TDI': 1812, 'Volkswagen Passat Highline DSG': 1813, 'Volkswagen Polo 1.0 MPI Comfortline': 1814, 'Volkswagen Polo 1.0 MPI Trendline': 1815, 'Volkswagen Polo 1.2 MPI Comfortline': 1816, 'Volkswagen Polo 1.2 MPI Highline': 1817, 'Volkswagen Polo 1.2 MPI Highline Plus': 1818, 'Volkswagen Polo 1.2 MPI Trendline': 1819, 'Volkswagen Polo 1.5 TDI Comfortline': 1820, 'Volkswagen Polo 1.5 TDI Highline': 1821, 'Volkswagen Polo 1.5 TDI Trendline': 1822, 'Volkswagen Polo Diesel Comfortline 1.2L': 1823, 'Volkswagen Polo Diesel Highline 1.2L': 1824, 'Volkswagen Polo Diesel Trendline 1.2L': 1825, 'Volkswagen Polo GT 1.5 TDI': 1826, 'Volkswagen Polo GT TDI': 1827, 'Volkswagen Polo GT TSI': 1828, 'Volkswagen Polo GTI': 1829, 'Volkswagen Polo IPL II 1.2 Petrol Highline': 1830, 'Volkswagen Polo Petrol Comfortline 1.2L': 1831, 'Volkswagen Polo Petrol Highline 1.2L': 1832, 'Volkswagen Polo Petrol Highline 1.6L': 1833, 'Volkswagen Polo Petrol Trendline 1.2L': 1834, 'Volkswagen Tiguan 2.0 TDI Highline': 1835, 'Volkswagen Vento 1.2 TSI Highline AT': 1836, 'Volkswagen Vento 1.5 Highline Plus AT 16 Alloy': 1837, 'Volkswagen Vento 1.5 TDI Comfortline': 1838, 'Volkswagen Vento 1.5 TDI Comfortline AT': 1839, 'Volkswagen Vento 1.5 TDI Highline': 1840, 'Volkswagen Vento 1.5 TDI Highline AT': 1841, 'Volkswagen Vento 1.5 TDI Trendline': 1842, 'Volkswagen Vento 1.6 Comfortline': 1843, 'Volkswagen Vento 1.6 Highline': 1844, 'Volkswagen Vento 2013-2015 1.6 Comfortline': 1845, 'Volkswagen Vento Diesel Breeze': 1846, 'Volkswagen Vento Diesel Comfortline': 1847, 'Volkswagen Vento Diesel Highline': 1848, 'Volkswagen Vento Diesel Trendline': 1849, 'Volkswagen Vento IPL II Petrol Highline AT': 1850, 'Volkswagen Vento IPL II Petrol Trendline': 1851, 'Volkswagen Vento Konekt Diesel Highline': 1852, 'Volkswagen Vento Magnific 1.6 Comfortline': 1853, 'Volkswagen Vento Petrol Comfortline': 1854, 'Volkswagen Vento Petrol Highline': 1855, 'Volkswagen Vento Petrol Highline AT': 1856, 'Volkswagen Vento Petrol Trendline': 1857, 'Volkswagen Vento Sport 1.2 TSI AT': 1858, 'Volkswagen Vento TSI': 1859, 'Volvo S60 D3': 1860, 'Volvo S60 D4 KINETIC': 1861, 'Volvo S60 D4 Momentum': 1862, 'Volvo S60 D4 SUMMUM': 1863, 'Volvo S60 D5 Summum': 1864, 'Volvo S80 2006-2013 D5': 1865, 'Volvo S80 D5': 1866, 'Volvo V40 Cross Country D3': 1867, 'Volvo V40 D3': 1868, 'Volvo V40 D3 R Design': 1869, 'Volvo XC60 D4 SUMMUM': 1870, 'Volvo XC60 D4 Summum': 1871, 'Volvo XC60 D5': 1872, 'Volvo XC60 D5 Inscription': 1873, 'Volvo XC90 2007-2015 D5 AT AWD': 1874, 'Volvo XC90 2007-2015 D5 AWD': 1875}, 'Location': {'Ahmedabad': 0, 'Bangalore': 1, 'Chennai': 2, 'Coimbatore': 3, 'Delhi': 4, 'Hyderabad': 5, 'Jaipur': 6, 'Kochi': 7, 'Kolkata': 8, 'Mumbai': 9, 'Pune': 10}, 'Fuel_Type': {'CNG': 0, 'Diesel': 1, 'Electric': 2, 'LPG': 3, 'Petrol': 4}, 'Transmission': {'Automatic': 0, 'Manual': 1}, 'Owner_Type': {'First': 0, 'Fourth & Above': 1, 'Second': 2, 'Third': 3}}\n"
     ]
    }
   ],
   "source": [
    "df.drop('New_Price',axis=1,inplace=True)\n",
    "from sklearn import preprocessing \n",
    "mapping_dict ={} \n",
    "category_col=['Name','Location','Fuel_Type','Transmission','Owner_Type']\n",
    "labelEncoder = preprocessing.LabelEncoder() \n",
    "for col in category_col: \n",
    "    df[col] = labelEncoder.fit_transform(df[col]) \n",
    "  \n",
    "    le_name_mapping = dict(zip(labelEncoder.classes_, \n",
    "                        labelEncoder.transform(labelEncoder.classes_))) \n",
    "  \n",
    "    mapping_dict[col]= le_name_mapping \n",
    "print(mapping_dict) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 2, 1, 3])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Owner_Type'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6017, 12)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "i=df[df['Mileage'].isnull()==True].index\n",
    "df.drop(i,inplace=True)\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5874, 12)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "j=df[df['Power'].isnull()==True].index\n",
    "df.drop(j,inplace=True)\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Name                 0\n",
       "Location             0\n",
       "Year                 0\n",
       "Kilometers_Driven    0\n",
       "Fuel_Type            0\n",
       "Transmission         0\n",
       "Owner_Type           0\n",
       "Mileage              0\n",
       "Engine               0\n",
       "Power                0\n",
       "Seats                2\n",
       "Price                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Power']=df.Power.astype(int)\n",
    "df['Engine']=df.Engine.astype(int)\n",
    "df['Mileage']=df.Mileage.astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 5874 entries, 0 to 6018\n",
      "Data columns (total 12 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Name               5874 non-null   int32  \n",
      " 1   Location           5874 non-null   int32  \n",
      " 2   Year               5874 non-null   int64  \n",
      " 3   Kilometers_Driven  5874 non-null   int64  \n",
      " 4   Fuel_Type          5874 non-null   int32  \n",
      " 5   Transmission       5874 non-null   int32  \n",
      " 6   Owner_Type         5874 non-null   int32  \n",
      " 7   Mileage            5874 non-null   int32  \n",
      " 8   Engine             5874 non-null   int32  \n",
      " 9   Power              5874 non-null   int32  \n",
      " 10  Seats              5872 non-null   float64\n",
      " 11  Price              5874 non-null   float64\n",
      "dtypes: float64(2), int32(8), int64(2)\n",
      "memory usage: 413.0 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Name</th>\n",
       "      <th>Location</th>\n",
       "      <th>Year</th>\n",
       "      <th>Kilometers_Driven</th>\n",
       "      <th>Fuel_Type</th>\n",
       "      <th>Transmission</th>\n",
       "      <th>Owner_Type</th>\n",
       "      <th>Mileage</th>\n",
       "      <th>Engine</th>\n",
       "      <th>Power</th>\n",
       "      <th>Seats</th>\n",
       "      <th>Price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1200</td>\n",
       "      <td>9</td>\n",
       "      <td>2010</td>\n",
       "      <td>72000</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>26</td>\n",
       "      <td>998</td>\n",
       "      <td>58</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>512</td>\n",
       "      <td>10</td>\n",
       "      <td>2015</td>\n",
       "      <td>41000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>19</td>\n",
       "      <td>1582</td>\n",
       "      <td>126</td>\n",
       "      <td>5.0</td>\n",
       "      <td>12.50</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>486</td>\n",
       "      <td>2</td>\n",
       "      <td>2011</td>\n",
       "      <td>46000</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>18</td>\n",
       "      <td>1199</td>\n",
       "      <td>88</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4.50</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1059</td>\n",
       "      <td>2</td>\n",
       "      <td>2012</td>\n",
       "      <td>87000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>20</td>\n",
       "      <td>1248</td>\n",
       "      <td>88</td>\n",
       "      <td>7.0</td>\n",
       "      <td>6.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>23</td>\n",
       "      <td>3</td>\n",
       "      <td>2013</td>\n",
       "      <td>40670</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>15</td>\n",
       "      <td>1968</td>\n",
       "      <td>140</td>\n",
       "      <td>5.0</td>\n",
       "      <td>17.74</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6014</th>\n",
       "      <td>1159</td>\n",
       "      <td>4</td>\n",
       "      <td>2014</td>\n",
       "      <td>27365</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>28</td>\n",
       "      <td>1248</td>\n",
       "      <td>74</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4.75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6015</th>\n",
       "      <td>668</td>\n",
       "      <td>6</td>\n",
       "      <td>2015</td>\n",
       "      <td>100000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>24</td>\n",
       "      <td>1120</td>\n",
       "      <td>71</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6016</th>\n",
       "      <td>932</td>\n",
       "      <td>6</td>\n",
       "      <td>2012</td>\n",
       "      <td>55000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>14</td>\n",
       "      <td>2498</td>\n",
       "      <td>112</td>\n",
       "      <td>8.0</td>\n",
       "      <td>2.90</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6017</th>\n",
       "      <td>1207</td>\n",
       "      <td>8</td>\n",
       "      <td>2013</td>\n",
       "      <td>46000</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>18</td>\n",
       "      <td>998</td>\n",
       "      <td>67</td>\n",
       "      <td>5.0</td>\n",
       "      <td>2.65</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6018</th>\n",
       "      <td>165</td>\n",
       "      <td>5</td>\n",
       "      <td>2011</td>\n",
       "      <td>47000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>25</td>\n",
       "      <td>936</td>\n",
       "      <td>57</td>\n",
       "      <td>5.0</td>\n",
       "      <td>2.50</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5874 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Name  Location  Year  Kilometers_Driven  Fuel_Type  Transmission  \\\n",
       "0     1200         9  2010              72000          0             1   \n",
       "1      512        10  2015              41000          1             1   \n",
       "2      486         2  2011              46000          4             1   \n",
       "3     1059         2  2012              87000          1             1   \n",
       "4       23         3  2013              40670          1             0   \n",
       "...    ...       ...   ...                ...        ...           ...   \n",
       "6014  1159         4  2014              27365          1             1   \n",
       "6015   668         6  2015             100000          1             1   \n",
       "6016   932         6  2012              55000          1             1   \n",
       "6017  1207         8  2013              46000          4             1   \n",
       "6018   165         5  2011              47000          1             1   \n",
       "\n",
       "      Owner_Type  Mileage  Engine  Power  Seats  Price  \n",
       "0              0       26     998     58    5.0   1.75  \n",
       "1              0       19    1582    126    5.0  12.50  \n",
       "2              0       18    1199     88    5.0   4.50  \n",
       "3              0       20    1248     88    7.0   6.00  \n",
       "4              2       15    1968    140    5.0  17.74  \n",
       "...          ...      ...     ...    ...    ...    ...  \n",
       "6014           0       28    1248     74    5.0   4.75  \n",
       "6015           0       24    1120     71    5.0   4.00  \n",
       "6016           2       14    2498    112    8.0   2.90  \n",
       "6017           0       18     998     67    5.0   2.65  \n",
       "6018           0       25     936     57    5.0   2.50  \n",
       "\n",
       "[5874 rows x 12 columns]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = df.iloc[:, :-1].values\n",
    "y = df.iloc[:, -1].values\n",
    "X\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')\n",
    "imputer.fit(X[:, 1:])\n",
    "X[:, 1:] = imputer.transform(X[:, 1:])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "sns.heatmap(df.corr(),annot=True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1707.,    7., 2016., ..., 1364.,   67.,    5.],\n",
       "       [ 634.,    8., 2014., ..., 1582.,  126.,    5.],\n",
       "       [1417.,    9., 2008., ..., 4806.,  500.,    5.],\n",
       "       ...,\n",
       "       [ 715.,    6., 2017., ..., 1197.,   81.,    5.],\n",
       "       [1546.,    5., 2010., ..., 3597.,  262.,    5.],\n",
       "       [1294.,    9., 2017., ..., 2987.,  254.,    5.]])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)\n",
    "X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor(n_estimators=10, random_state=0)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)\n",
    "regressor.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([50.414])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "regressor.score(X_test,y_test)\n",
    "regressor.predict([[8,1,2017,40000,1,0,2,15,9980,10000,5]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[10.95  5.7 ]\n",
      " [27.99 30.  ]\n",
      " [10.62 10.  ]\n",
      " ...\n",
      " [ 8.62  9.8 ]\n",
      " [ 7.44  3.5 ]\n",
      " [12.7  10.55]]\n"
     ]
    }
   ],
   "source": [
    "y_pred = regressor.predict(X_test)\n",
    "np.set_printoptions(precision=2)\n",
    "print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(regressor,open('model.pkl','wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

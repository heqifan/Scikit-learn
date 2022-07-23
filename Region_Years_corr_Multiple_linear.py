# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:10:04 2022

@author: HYF
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:19:24 2022

@author: HYF
"""
import statsmodels.api as sm
import pingouin as pg
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys,glob
from osgeo import osr,ogr,gdal
import scipy.interpolate as spi
from scipy.stats import linregress
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as plt 
from scipy import signal
from sklearn.preprocessing import StandardScaler
INPATH = r'D:\Cheng'
OUTPATH = r'D:\Cheng\Result'
RCPs = ['RCP45','RCP85']
Vars = ['npp','prc','tas']
Var_names = ['降水','气温']
sort = ['青藏高原区','热带-亚热带季风区','温带季风区','温带大陆区','全国陆地']
Sample_tif = r'D:\Cheng\Climate_4R_CEVSA.tif'
breakpoint_path = r'D:\Cheng\PT_CEVSA_NPP0721.xls'
styear = 2006
edyear = 2099
interval = 20
all_years = [year for year in range(styear,edyear+1)]

def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    
    return x
 
def plot_trendline(x1,x2,y1,y2,car1,car2,x_label,y_label,x_lim,y_lim,x_ticks,y_ticks,title,fontsize,n,inpath):
    '''
    Args:
        input:
            x1:  第一条线的x数组
            x2:  第二条线的x数组
            y1： 第一条线的y数组
            y2：第二条线的y数组
            car1： 第一条线的图例
            car2： 第二条线的图例
            x_label：x轴上的标签
            y_label: y轴上的标签
            x_lim: x轴上的长度区间
            y_lim: y轴上的长度区间
            x_ticks：x轴上的刻度数组
            y_ticks：y轴上的刻度数组
            title:  图例的标题
            fontsize:  字体的大小
            n:  函数的次数
            inpath:  输入的路径
            
    '''
    mpl.pylab.plot(x1, y1, 'bo')
    parameter = np.polyfit(x1, y1, n) # n=1为一次函数，返回函数参数
    f = np.poly1d(parameter) # 拼接方程
    mpl.pylab.plot(x1, f(x1),"b-",label=car1)
    
    mpl.pylab.plot(x2, y2,  'ro')
    parameter = np.polyfit(x2, y2, n) # n=1为一次函数，返回函数参数
    f = np.poly1d(parameter) # 拼接方程
    mpl.pylab.plot(x2, f(x2),"r-",label=car2)
    
    mpl.pylab.xlabel(x_label)             # x、y轴的介绍
    mpl.pylab.ylabel(y_label)
    
    mpl.pylab.xlim(x_lim)            # x、y轴的长度区间
    #mpl.pylab.ylim([0, 1])
    
    mpl.pylab.title(title)
  
    mpl.pylab.xticks(x_ticks, fontsize=fontsize)       # x、y轴的刻度
    #mpl.pylab.yticks(y_ticks, fontsize=8)
    
    mpl.pylab.legend(loc="upper right")
    
    mpl.pylab.savefig(inpath, dpi=600)
    
    plt.show()

    plt.close()


def Get_tif_xy(Sample_tif):
    '''
    Args: 
    input:
        Sample_tif:  输入的tif数据的路径
    return:
        arr_x:   tif数据中像元对应的经度的数组
        arr_y:   tif数据中像元对应的纬度的数组
        nXSize： tif数据的列数
        nYSize： tif数据的行数
    '''
    
    dataset = gdal.Open(Sample_tif)  # 打开tif
    
    adfGeoTransform = dataset.GetGeoTransform()  # 读取地理信息
    
    # 左上角地理坐标
    print('左上角x地理坐标：',adfGeoTransform[0])
    print('左上角y地理坐标：',adfGeoTransform[3])
    
    nXSize = dataset.RasterXSize  # 列数
    nYSize = dataset.RasterYSize  # 行数
    
    print('列数为：',nXSize, '行数为：',nYSize)
    
    arr_x = []  # 用于存储每个像素的（X，Y）坐标
    arr_y = []
    for i in range(nYSize):
        row_x = []
        row_y = []
        for j in range(nXSize):
            px = adfGeoTransform[0] + j * adfGeoTransform[1] + i * adfGeoTransform[2]
            py = adfGeoTransform[3] + j * adfGeoTransform[4] + i * adfGeoTransform[5]
            row_x.append(px)
            row_y.append(py)
            #print(px,py)
        arr_x.append(row_x)
        arr_y.append(row_y)
        
    return arr_x,arr_y,nXSize,nYSize


def read_img(filename):
    '''
    Args:
    input:
        filename: 输入的tif数据的路径
    output:
        im_data: tif数据对应的数组
    '''
    dataset=gdal.Open(filename)       #打开文件
 
    im_width = dataset.RasterXSize    #栅格矩阵的列数
    im_height = dataset.RasterYSize   #栅格矩阵的行数
 
    im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
 
    del dataset 
    return im_data
    
def dt(data_mean):
    
    '''去趋势'''
    data_detrend = signal.detrend(data_mean)
    
    return data_detrend


def dm(data):
    '''去掉无效值'''
    data.dropna(axis=0, how='any', inplace=True)
    
    '''计算每个分区每年的平均值'''
    data_mean = np.nanmean(np.array(data),axis = 0)
    
    return data_mean

def Standard(data_detrend):
    
    '''标准化'''
    data_Standard = ZscoreNormalization(data_detrend)
    
    return data_Standard

All_start_time = datetime.datetime.now()
'''获取分区的经纬度'''
'''x是经度，列   y是纬度，行'''
Region_x,Region_y,col,line = Get_tif_xy(Sample_tif)   #获取经度，纬度
# print('tiff数据的前五行的经度为：',Region_x[0:5])
# print('tiff数据的前五行的纬度为：',Region_y[0:5])
Region_x_array,Region_y_array = np.array(Region_x).flatten(),np.array(Region_y).flatten()  #将存放经纬度的二维列表转为一维数组
Region_x_y_array = pd.DataFrame(np.array([Region_x_array,Region_y_array]).T).astype('str')   #将两个一维数组合并并倒置转为数据框
Region_x_y_array[0] = Region_x_y_array[0].apply(lambda x:x.split('.')[0] + '.' + x.split('.')[1][0])
Region_x_y_array[1] = Region_x_y_array[1].apply(lambda x:x.split('.')[0] + '.' + x.split('.')[1][0])   #更改数据类型
Region_x_y_array = Region_x_y_array.rename(columns = {0:'经度',1:'纬度'})   #重命名列名
Region_x_y_array['Lon_Lat'] = Region_x_y_array['经度'] + '_' + Region_x_y_array['纬度']   #添加一列的值为两列相加
Region_data = read_img(Sample_tif)    #读取tif数据的值
Region_x_y_array['Region_value'] = Region_data.flatten().astype('int')   #添加一列值

'''获取真实数据值的经纬度'''
ppth = INPATH + os.sep + '2007.txt'     
data = pd.read_csv(ppth,sep='\s+',header=None,index_col = False)
lon,lat = data.iloc[:,0],data.iloc[:,1]
x_y_array = pd.DataFrame(np.array([lon,lat]).T).rename(columns = {0:'经度',1:'纬度'})
x_y_array['经度'] = x_y_array['经度'].apply(lambda x:round(x,1))
x_y_array['纬度'] = x_y_array['纬度'].apply(lambda x:round(x,1))
x_y_array['Lon_Lat'] = x_y_array['经度'].astype('str') + '_' + x_y_array['纬度'].astype('str')   

'''读取断点数据'''
points_data = pd.read_excel(breakpoint_path)
points_data['V3'] = points_data['V3'].apply(lambda x:round(x))

rcp45_npp = list(points_data.loc[(points_data['情景']==1) & (points_data['npp prc tas wi vpd']==1)]['V3'])
rcp45_prc = list(points_data.loc[(points_data['情景']==1) & (points_data['npp prc tas wi vpd']==2)]['V3'])
rcp45_tas = list(points_data.loc[(points_data['情景']==1) & (points_data['npp prc tas wi vpd']==3)]['V3'])

rcp85_npp = list(points_data.loc[(points_data['情景']==2) & (points_data['npp prc tas wi vpd']==1)]['V3'])
rcp85_prc = list(points_data.loc[(points_data['情景']==2) & (points_data['npp prc tas wi vpd']==2)]['V3'])
rcp85_tas = list(points_data.loc[(points_data['情景']==2) & (points_data['npp prc tas wi vpd']==3)]['V3'])

points = {'RCP45':[rcp45_npp,rcp45_prc,rcp45_tas],'RCP85':[rcp85_npp,rcp85_prc,rcp85_tas]}

'''读取npp数据'''
for year in range(styear,edyear+1): #计算最后一年的年份
    #print(year,year+10)
    if year+10>edyear:
        eddyear = year-1
        print(f'最后一年为：{year-1}')
        break

    
for rcp in RCPs:   #循环两种数据类型
    print('rcp：',rcp)
    '''将每个变量的所有年都存放在一起'''
    npp_year_data,prc_year_data,tas_year_data = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    npp_year_data['Lon_Lat'],prc_year_data['Lon_Lat'],tas_year_data['Lon_Lat'] = x_y_array['Lon_Lat'],x_y_array['Lon_Lat'],x_y_array['Lon_Lat']
    for num,var in enumerate([npp_year_data,prc_year_data,tas_year_data]):   #循环三种变量
        for year in range(styear,edyear+1):   #循环每一年的数据
            print(f'开始存放{rcp}-{Vars[num]}-{year}的数据')
            in_data =  INPATH + os.sep + rcp + os.sep + Vars[num] +  os.sep + str(year) + '.txt'
            try:
                data=pd.read_csv(in_data,sep='\s+',header=None,index_col = False).astype('float')  
            except:
                print('有问题的数据为：',in_data)
                continue
            if Vars[num] == 'npp':
                var[year] = np.sum(data.iloc[:,69:105], axis=1)   #npp求和
            elif Vars[num] == 'prc':
                var[year] = np.sum(data.iloc[:,2:], axis=1)   #prc求和
            else:
                var[year] = np.average(data.iloc[:,2:], axis=1)  #tas求平均
    print('所有年份数据已经存放完毕')   
    
    '''将对应分区编号与实际数值根据经纬度进行连接'''         
    npp_join_data = pd.merge(Region_x_y_array,npp_year_data,on = 'Lon_Lat',how = 'left')   
    prc_join_data = pd.merge(Region_x_y_array,prc_year_data,on = 'Lon_Lat',how = 'left')
    tas_join_data = pd.merge(Region_x_y_array,tas_year_data,on = 'Lon_Lat',how = 'left')
    
    '''定义存放每个分区的数据'''
    value_data1,value_data2,value_data3,value_data4,value_data5 = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()  
    
    '''定义存放所有分区的数据'''
    all_corr_data = pd.DataFrame()
    all_regession_data = pd.DataFrame()
    all_mean_data = pd.DataFrame()
    all_dt_data = pd.DataFrame()
    all_st_data = pd.DataFrame()
    '''循环每个分区数据'''
    for num,value_data in enumerate([value_data1,value_data2,value_data3,value_data4,value_data5]):#循环每个分区的数据
        '''对应年数据提取出来'''
        for year in range(styear,eddyear+1):  #循环每年
            all_yearss = [year2 for year2 in range(year,year+11)]
            if num==4:  #如果循环到全国分区的话
                npp = npp_join_data[npp_join_data['Region_value'].isin([1,2,3,4])].loc[:,all_yearss]
                prc = prc_join_data[prc_join_data['Region_value'].isin([1,2,3,4])].loc[:,all_yearss]
                tas = tas_join_data[tas_join_data['Region_value'].isin([1,2,3,4])].loc[:,all_yearss]
            else:  #如果还没循环到全国分区的话
                npp = npp_join_data[npp_join_data['Region_value']==num+1].loc[:,all_yearss]
                prc = prc_join_data[prc_join_data['Region_value']==num+1].loc[:,all_yearss]
                tas = tas_join_data[tas_join_data['Region_value']==num+1].loc[:,all_yearss]            
            npp.dropna(axis=0, how='any', inplace=True)
            prc.dropna(axis=0, how='any', inplace=True)
            tas.dropna(axis=0, how='any', inplace=True)
            
            '''计算每10年的变异系数'''
            npp_cv = ((np.nanstd((np.nanmean(np.array(npp),axis = 0))))/(np.nanmean((np.nanmean(np.array(npp),axis = 0))))) 
            prc_cv = ((np.nanstd((np.nanmean(np.array(prc),axis = 0))))/(np.nanmean((np.nanmean(np.array(prc),axis = 0))))) 
            tas_cv = ((np.nanstd((np.nanmean(np.array(tas),axis = 0))))/(np.nanmean((np.nanmean(np.array(tas),axis = 0))))) 
            
            '''存放每10年的变异系数'''
            value_data[year] = np.array([npp_cv,prc_cv,tas_cv])
        value_data.index = ['npp_cv','prc_cv','tas_cv']
        '''偏相关分析'''
        data_prc = pg.partial_corr(value_data.T, 'prc_cv', 'npp_cv', 'tas_cv')
        data_tas = pg.partial_corr(value_data.T, 'tas_cv', 'npp_cv','prc_cv')
        print(f'prc_corr: {data_prc}')
        print(f'tas_corr: {data_tas}')
        all_corr_data[str(sort[num])] = np.array([data_prc['r'][0]**2,data_prc['r'][0],data_prc['p-val'][0],data_tas['r'][0]**2,data_tas['r'][0],data_tas['p-val'][0]]) 
        npp_fore_years = [year for year in range(styear,points[rcp][0][num]+1)]
        npp_behead_years =  [year for year in range(points[rcp][0][num],edyear+1)]
        
        if num==4:  #如果循环到全国分区的话
            npp_1 = npp_join_data[npp_join_data['Region_value'].isin([1,2,3,4])].loc[:,npp_fore_years]
            npp_2 = npp_join_data[npp_join_data['Region_value'].isin([1,2,3,4])].loc[:,npp_behead_years]
            prc_ = prc_join_data[prc_join_data['Region_value'].isin([1,2,3,4])].loc[:,all_years]
            tas_ = tas_join_data[tas_join_data['Region_value'].isin([1,2,3,4])].loc[:,all_years]
            
        else :  #如果还没循环到全国分区的话
            npp_1 = npp_join_data[npp_join_data['Region_value']==num+1].loc[:,npp_fore_years]
            npp_2 = npp_join_data[npp_join_data['Region_value']==num+1].loc[:,npp_behead_years]
            prc_ = prc_join_data[prc_join_data['Region_value']==num+1].loc[:,all_years]
            tas_ = tas_join_data[tas_join_data['Region_value']==num+1].loc[:,all_years]
            
        '''清洗数据-平均'''
        npp_1_mean,npp_2_mean,prc_mean,tas_mean = dm(npp_1),dm(npp_2),dm(prc_),dm(tas_)
        npp_mean = np.append(npp_1_mean[:-1],npp_2_mean)
        all_mean_data['npp'+ sort[num]] = npp_mean
        all_mean_data['prc'+ sort[num]] = prc_mean
        all_mean_data['tas'+ sort[num]] = tas_mean
        
        '''去趋势-合并'''
        npp_1_dt,npp_2_dt,prc_dt,tas_dt = dt(npp_1_mean),dt(npp_2_mean),dt(prc_mean),dt(tas_mean)
        npp_dt = np.append(npp_1_dt[:-1],npp_2_dt)
        all_dt_data['npp'+ sort[num]] = npp_dt
        all_dt_data['npp'+ sort[num]] = prc_dt
        all_dt_data['npp'+ sort[num]] = tas_dt 
        print('npp长度为：',npp_dt.shape)
        print('prc长度为：',prc_dt.shape)
        print('tas长度为：',tas_dt.shape)
        
        '''标准化'''
        npp_,prc_,tas_ = Standard(npp_dt),Standard(prc_dt),Standard(tas_dt)
        all_st_data['npp'+ sort[num]] = npp_
        all_st_data['npp'+ sort[num]] = prc_
        all_st_data['npp'+ sort[num]] = tas_
        
        '''多元线性回归'''
        result_ = sm.OLS(npp_,sm.add_constant(pd.DataFrame({'x1': prc_, 'x2': tas_ }))).fit()
        coef_x1 = result_.params[1]
        coef_x2 = result_.params[2]
        print(result_.summary())
        print('coef:',result_.params)

        all_regession_data[sort[num]] = np.array([coef_x1,coef_x2,result_.f_pvalue,result_.rsquared])
        
    all_mean_data.index = all_years
    all_mean_data.to_excel(OUTPATH + os.sep + 'Result_' + rcp + os.sep +'Multiple_linear' + os.sep  + rcp + '_ALL_Year_data.xlsx', sheet_name= 'ALL_Year_data')
    print('\n年值数据为：',all_mean_data)
    
    all_dt_data.index = all_years
    all_dt_data.to_excel(OUTPATH + os.sep + 'Result_' + rcp + os.sep  +'Multiple_linear' + os.sep  + rcp + '_ALL_Detrend_data.xlsx', sheet_name= 'ALL_Detrend_data')
    print('\n去趋势之后的数据为：',all_dt_data)
    
    all_st_data.index = all_years
    all_st_data.to_excel(OUTPATH + os.sep + 'Result_' + rcp + os.sep +'Multiple_linear' + os.sep  + rcp + '_ALL_Standard_data.xlsx', sheet_name= 'ALL_Standard_data')
    print('\n标准化之后的数据为：',all_st_data)
    
    all_regession_data.index = ['prc_coef','tas_coef','P','R2']
    all_regession_data.T.to_excel(OUTPATH + os.sep + 'Result_' + rcp + os.sep + 'Multiple_linear' + os.sep  + rcp + '_ALL_Region_regession.xlsx', sheet_name= 'ALL_Region_regession')
    print('\n多元线性回归后的数据为的数据为：',all_regession_data.T)
    
    all_corr_data.index = ['prc_R2','prc_coef','prc_P','tas_R2','tas_coef','tas_P']
    all_corr_data.T.to_excel(OUTPATH + os.sep+ 'Result_' + rcp + os.sep + 'Partial_corr' + os.sep + rcp + '_ALL_Region_corr.xlsx', sheet_name= 'ALL_Region_corr')
    print('\n偏相关分析后的数据为的数据为：',all_corr_data.T)
    
    print(f'-------{rcp} is ok----------')
    
    
All_end_time = datetime.datetime.now()
print('\n All   is    ok  !!!!!!!')
print(f'花费总时间为： {All_end_time -All_start_time}')
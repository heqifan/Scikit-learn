# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:19:24 2022

@author: HYF
"""

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
INPATH = r'D:\Cheng'
OUTPATH = r'D:\Cheng\Result'
RCPs = ['RCP45','RCP85']
Vars = ['clo','hum','prc','tas']
Var_names = ['云量','相对湿度','降水','气温']
sort = ['青藏高原区','热带-亚热带季风区','温带季风区','温带大陆区','全国陆地']
Sample_tif = r'D:\Cheng\Climate_4R_CEVSA.tif'
styear = 2006
edyear = 2099
interval = 20

 
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
    
    
    
    
'''获取分区的经纬度'''
'''x是经度，列   y是纬度，行'''
Region_x,Region_y,col,line = Get_tif_xy(Sample_tif)   #获取经度，纬度
print('tiff数据的前五行的经度为：',Region_x[0:5])
print('tiff数据的前五行的纬度为：',Region_y[0:5])
Region_x_array,Region_y_array = np.array(Region_x).flatten(),np.array(Region_y).flatten()  #将存放经纬度的二维列表转为一维数组
Region_x_y_array1 = pd.DataFrame(np.array([Region_x_array,Region_y_array]).T).astype('str')   #将两个一维数组合并并倒置转为数据框
Region_x_y_array = Region_x_y_array1.copy()
Region_x_y_array[0] = Region_x_y_array1[0].apply(lambda x:x.split('.')[0] + '.' + x.split('.')[1][0])
Region_x_y_array[1] = Region_x_y_array1[1].apply(lambda x:x.split('.')[0] + '.' + x.split('.')[1][0])   #更改数据类型
Region_x_y_array = Region_x_y_array.rename(columns = {0:'经度',1:'纬度'})   #重命名列名
Region_x_y_array['Lon_Lat'] = Region_x_y_array['经度'] + '_' + Region_x_y_array['纬度']   #添加一列的值为两列相加
Region_data = read_img(Sample_tif)    #读取tif数据的值
Region_x_y_array['Region_value'] = Region_data.flatten().astype('int')   #添加一列值


'''获取真实的像元面积'''
area_path = INPATH + os.sep + '2005.csv'    
area_data =pd.read_csv(area_path,sep=',',header=None,index_col = False)
area_ = area_data.iloc[:,0]



'''获取真实数据值的经纬度'''
ppth = INPATH + os.sep + '2007.txt'     
data=pd.read_csv(ppth,sep='\s+',header=None,index_col = False)
lon,lat = data.iloc[:,0],data.iloc[:,1]
x_y_array = pd.DataFrame(np.array([lon,lat]).T).rename(columns = {0:'经度',1:'纬度'})
x_y_array['经度'] = x_y_array['经度'].apply(lambda x:round(x,1))
x_y_array['纬度'] = x_y_array['纬度'].apply(lambda x:round(x,1))
x_y_array['Lon_Lat'] = x_y_array['经度'].astype('str') + '_' + x_y_array['纬度'].astype('str')   
x_y_array['C_area'] = area_.astype('float')


i= 0

All_start_time = datetime.datetime.now()
for var in Vars:   #循环四个变量
    start_time = datetime.datetime.now()
    Rcp_45_years = []  #定义两个空的列表用于存放两种数据的五个分区的数据
    Rcp_85_years = []  
    for rcp in RCPs:   #循环两种数据
        var_allyear =  pd.DataFrame()   #定义空的数据框存放每年全国的数据
        '''将每个变量每钟数据的所有年存储起来'''
        for year in range(styear,edyear+1):   #定义年份的循环
            print(year,end = ' ')
            inpath = INPATH + os.sep + rcp + os.sep + var + os.sep + str(year) + '.txt'   #输入数据
            try:
                data=pd.read_csv(inpath,sep='\s+',header=None,index_col = False).astype('float')  #读取数据
            except:
                print('\n——————————————————————')
                print('有问题的数据为：',inpath)
                print('\n——————————————————————')
                continue
            Data = data.iloc[:,2:]   #获取每年的数据  
            if var == 'prc':
                cal = np.sum(Data, axis=1)   #按行求平均
            else:
                cal = np.average(Data, axis=1)   #按行求和
            var_allyear[year] = cal  #将数据存入大的数据框
        value_data1,value_data2,value_data3,value_data4,value_data5 = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()  #定义数据框存放
        
        for num,value_data in enumerate([value_data1,value_data2,value_data3,value_data4,value_data5]):#循环每个分区的数据
            '''将每种变量每种数据的每个分区所有年数据提取出来'''
            asp = num+1
            for year2 in range(styear,edyear+1):  #循环每年
                print(year2,end = ' ')
                x_y_array['Value'] = var_allyear[year2]#将每年的数据依次赋值
                join_data = pd.merge(Region_x_y_array,x_y_array,on = 'Lon_Lat',how = 'left')   #将大的数据框与实际数组数据框进行连接
                if num==4:  #如果循环到全国分区的话
                    value_data[year2] = join_data[join_data['Region_value'].isin([1,2,3,4])]['Value'].astype('float') 
                    value_data['C_area'] = join_data[join_data['Region_value'].isin([1,2,3,4])]['C_area'].astype('float') 
                else:  #如果还没循环到全国分区的话
                    # value_data[[year2,'C_area']] = join_data[join_data['Region_value']==asp][['Value','C_area']].astype('float') 
                    value_data[year2] = join_data[join_data['Region_value']==asp]['Value'].astype('float') 
                    value_data['C_area'] = join_data[join_data['Region_value']==asp]['C_area'].astype('float') 
                value_data['Area_Value_' + str(year2)] = value_data[year2].astype('float')*value_data['C_area'].astype('float') 
            m_s_c_data = value_data[['C_area']+['Area_Value_'+ str(year) for year in range(styear,edyear+1)]].astype('float')  #获取每年的原始数据
            year_data =  value_data[['C_area']+['Area_Value_'+ str(year) for year in range(styear,edyear+1)]].astype('float')  #获取每年乘以对应面积的
            #计算每个分区的mean(),std(),cv()
            print('\n m_s_c_data中空值的形状为：',m_s_c_data[m_s_c_data.isnull().T.any().T].shape)
            print('\n year_data中空值的形状为：',year_data[year_data.isnull().T.any().T].shape)
            
            m_s_c_data.dropna(axis=0, how='any', inplace=True)
            print('m_s_c_data.shape',m_s_c_data.shape)
            
            year_data.dropna(axis=0, how='any', inplace=True)
            print('year_data.shape',year_data.shape)
            
            mean_data = (np.nanmean(np.array(m_s_c_data.iloc[:,1:]),axis = 0))/(np.nansum(np.array(m_s_c_data['C_area']),axis = 0))   #分区求平均
            std_data = (np.nanstd(np.array(m_s_c_data.iloc[:,1:]),axis = 0))/(np.nansum(np.array(m_s_c_data['C_area']),axis = 0))     #分区求标准差
            cv_data = (std_data/mean_data)*100                      #分区求变异系数
            yy_data = (np.nansum(np.array(year_data.iloc[:,1:]),axis = 0))/(np.nansum(np.array(year_data['C_area']),axis = 0))  #分区求年值
            print('\n——————————————————————————————————————————')
            print(f'{var}————{rcp}————第{num+1}区域的值为{yy_data}')
            print('\n——————————————————————————————————————————')
            if rcp =='RCP45':  #如果是RCP45的话就合并
                Rcp_45_years.append(yy_data.tolist())
            else:   #如果是RCP85的话就合并
                Rcp_85_years.append(yy_data.tolist())
            #将要导出的数据转为数据框
            all_data = pd.DataFrame(np.array([yy_data,std_data,cv_data]).T,columns=["年值","标准差","变异系数"],index=[year for year in range(styear,edyear+1)])
            #写出
            # with pd.ExcelWriter(OUTPATH + os.sep+ 'Result_' + rcp + os.sep + 'Every_Year_Value_Std_CV.xlsx', engine='xlsxwriter') as writer1:   
            #     all_data.to_excel(writer1, sheet_name= var + '_'+ sort[num])
            #     writer1.save() 
            #     writer1.close() 
            all_data.to_excel(OUTPATH + os.sep+ 'Result_' + rcp + os.sep + 'Every_Year_Value_Std_CV'+ os.sep + var + '_'+ sort[num] + '.xlsx', sheet_name= var + '_'+ sort[num])
            print('\n' + OUTPATH + os.sep+ 'Result_' + rcp + os.sep + 'Every_Year_Value_Std_CV' + os.sep + var + '_'+ sort[num] + '.xlsx' + '  is  okkk!!!!!')
            #计算每个分区所有年的slope, intercept, r_value, p_value, std_err
            slope, intercept, r_value, p_value, std_err = linregress(np.array([year for year in range(styear,edyear+1)]),yy_data)    
            r_squared = r_value**2
            all_data2 = pd.DataFrame([[slope,r_squared,p_value]],columns=["Slope","R2","P"])
            # with pd.ExcelWriter(OUTPATH + os.sep+ 'Result_' + rcp + os.sep + 'Slope_R2_P.xlsx', engine='xlsxwriter') as writer2:
            #     all_data2.to_excel(writer2, sheet_name= var + '_'+ sort[num])
            #     writer2.save() 
            #     writer2.close()
            all_data2.to_excel(OUTPATH + os.sep+ 'Result_' + rcp + os.sep + 'Slope_R2_P' + os.sep + var + '_'+ sort[num] + '.xlsx', sheet_name= var + '_'+ sort[num])
            print('\n' + OUTPATH + os.sep+ 'Result_' + rcp + os.sep + 'Slope_R2_P' + os.sep + var + '_'+ sort[num] + '.xlsx' + '  is  okkk!!!!!')            
    for num in range(len(Rcp_45_years)):
        
        x1 = [year for year in range(styear,edyear+1)]  #年份
        x2 = x1.copy()
        
        y_45 = Rcp_45_years[num]
        y_85 = Rcp_85_years[num]
        
        x_label = '年'  #设置x标签
        y_label = Var_names[i]  #设置y轴标签
        
        title = sort[num]
        
        x_lim = np.array([2000, edyear+1])                   # x,y轴的长度区间
        y_lim = None
        
        x_ticks = np.arange(2000, edyear+1, interval)     # x,y的刻度区间
        y_ticks = None
        
        fontsize = 10   #字体大小
        
        car1 = 'RCP_45'  #图例
        car2 = 'RCP_85'   #图例
        
        inpath =  OUTPATH + os.sep + var + '_' + title + '.jpg'
        try:
            plot_trendline(x1,x2,y_45,y_85, car1,car2,x_label,y_label,x_lim,y_lim,x_ticks,y_ticks,title,fontsize,1,inpath)
        except:
            continue
    i+=1  
    end_time = datetime.datetime.now()
    print(f'\n {var} all   is    ok    !!!!! ')
    print(f'\n {var}花费时间为{end_time-start_time}')
All_end_time = datetime.datetime.now()

print('\n All   is    ok  !!!!!!!')
print(f'花费总时间为： {All_end_time -All_start_time}')
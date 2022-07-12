import streamlit as st
import numpy as np
import pandas as pd
from utils import _initialize_spark
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.functions import udf, col
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, BTRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

def tranformFetures(X, assembler):
    # Tạo bản sao để tránh ảnh hưởng dữ liệu gốc
    X_ = X.copy()
    assembled_X = assembler.transform(X_)
    
    return assembled_X
def prediction(samples, model):
    st.write("predict")
    # Encode dữ liệu
    X_scaled = tranformFetures(samples, assembler)
    # Predict
    return model.predict(X_scaled)
#l = list(range(10))
# st.write(l)

#rdd = sc.parallelize(l)
#rdd.cache()
#st.write(rdd)

#st.write("## Get results through actions")
#st.write(rdd.collect())
#st.write(rdd.take(3))
#st.write(rdd.count())

#st.write("## Transform RDDs")
#st.write(rdd.filter(lambda x: x%2==0).collect())  # talk about lazy evaluation here: filter still non evaluated rdd
#st.write(rdd.map(lambda x: x*2).collect()) 
#st.write(rdd.map(lambda x: x*2).reduce(lambda x, y: x + y))  # reduce runs all previous rdds
# Compare the two following
#st.write(rdd.map(lambda x: list(range(x))).collect()) 
#st.write(rdd.flatMap(lambda x: list(range(x))).collect()) 

#st.write("## Wordcount")
#file_rdd = sc.textFile("lorem.txt")
#st.write(file_rdd.collect())  # so what's inside ?
#st.write(file_rdd.flatMap(lambda sentence: sentence.split()).map(lambda word: (word, 1)).reduceByKey(lambda x,y: x+y).collect())
#st.write()

def LR_model(choice_input):
    st.subheader('Mô hình LR')
    if choice_input == 'Dữ liệu mẫu':
        st.write('#### Sample dataset', pd_df)

        # Chọn dữ liệu từ mẫu
        selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
        selected_rows = data.collect()[selected_indices[-1]]

        st.write('#### Kết quả')

        if st.button('Dự đoán'):
            if not selected_rows.empty:
                X = selected_rows
                pred = prediction(X, model_lr)

                # Xuất ra màn hình
                results = pd.DataFrame({'Giá dự đoán': pred,
                                        'Giá thực tế': selected_rows.death_rate})
                st.write(results)
            else:
                st.error('Hãy chọn dữ liệu trước')

    elif choice_input == 'Tự chọn':
        st.write('Coming soon')

def RF_model(choice_input):
    st.subheader('Mô hình RF')
    if choice_input == 'Dữ liệu mẫu':
        st.write('#### Sample dataset', pd_df)

        # Chọn dữ liệu từ mẫu
        selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
        #selected_rows = data.collect()[selected_indices[-1]]

        st.write('#### Kết quả')

        if st.button('Dự đoán'):
            if not selected_rows.empty:
                X = selected_rows
                pred = prediction(X, model_lr)

                # Xuất ra màn hình
                results = pd.DataFrame({'Giá dự đoán': pred,
                                        'Giá thực tế': selected_rows.death_rate})
                st.write(results)
            else:
                st.error('Hãy chọn dữ liệu trước')

    elif choice_input == 'Tự chọn':
        st.write('Coming soon')

def main():
    st.title('Dự đoán giá bất động sản')
    model_list = ['Mô hình Linear Regression',
                      'Mô hình Random Forest',
                      'Mô hình Gradient Boosted']
    choice_model = st.sidebar.selectbox('Mô hình huấn luyện trên:', model_list)
    #st.write("hello")
    input = ['Dữ liệu mẫu', 'Tự chọn']
    #st.write("hello")
    choice_input = st.sidebar.selectbox('Chọn kiểu nhập dữ liệu:', input)

    if choice_model == 'Mô hình Linear Regression':
        LR_model(choice_input)

    elif choice_model == 'Mô hình Random Forest':
        RF_model(choice_input)

    elif choice_model == 'Mô hình Gradient Boosted':
        RF_model(choice_input)

if __name__ == '__main__':
    spark, sc = _initialize_spark()
    ## Load dataset
    df = spark.read.format('org.apache.spark.sql.json').load("./clean/clean.json")
    #st.write("df ready")
    df=df.withColumnRenamed("P. sinh hoạt chung","Phòng sinh hoạt chung")
    df=df.withColumnRenamed("T.T thương mại tòa nhà","TT thương mại tòa nhà")
    df=df.withColumnRenamed("T.T thương mại","TT thương mại")
    df=df.withColumnRenamed("Bàn ghế P.Khách","Bàn ghế PKhách")
    #st.write("create cols")
    cols = ['Ban công riêng', 'Chỗ để ô tô', 'Cà phê', 'Công viên tòa nhà', 'Hầm để xe', 'Hồ bơi chung', 'Nhà hàng',
            'Phòng sinh hoạt chung', 'Phòng tập gym', 'siêu thị mini tòa nhà', 'Sân chơi trẻ em', 'Sân nướng BBQ',
            'TT thương mại tòa nhà', 'Bệnh viện', 'Cao tốc', 'Chỗ đậu ô tô', 'Chợ', 'Công viên', 'Cảng biển',
            'Gần biển', 'Gần núi', 'Gần rừng', 'Gần suối', 'Gần sông', 'Hồ bơi', 'Khu công nghiệp', 'Khu du lịch',
            'Khu dân cư', 'Nhà hát', 'Nhà thiếu nhi', 'Nhà văn hóa', 'Rạp chiếu phim', 'Siêu thị', 'Siêu thị mini',
            'Sân bay', 'Sân vận động', 'TT thương mại', 'Trường cấp 1', 'Trường cấp 2', 'Trường cấp 3', 'Trường mầm non',
            'Trường Đại học', 'Bàn ghế PKhách', 'Bàn phấn', 'Bàn ăn', 'Bình nóng lạnh', 'Bồn rửa mặt', 'Bồn tắm',
            'Dàn Karaoke', 'Giường ngủ', 'Kệ TV', 'Máy giặt', 'Máy hút mùi', 'Máy lạnh', 'Máy rửa chén', 'Tivi',
            'Tủ bếp', 'Tủ lạnh', 'Tủ quần áo', 'Tủ trang trí', 'Vòi hoa sen', 'Điện ba pha', 'Thân thiện', 'Trí thức',
            'Bảo vệ 24/24', 'Carmera an ninh', 'Đường bê tông', 'Đường trải nhựa', 'Đường trải đá', 'Đường đất']
    #st.write("have cols")
    for col_name in cols:
        df = df.withColumn(col_name, col(col_name).cast('int'))
    data = df.drop(*['TienIchGanDat','id','NgayDangBan', 'MoTa_Vec'])
    #st.write("data ready")
    pd_df = data.toPandas()
    features = data.columns
    features = [ele for ele in features if ele not in ['MaTin','TongGia','Gia/m2']]
    assembler = VectorAssembler(inputCols = features, outputCol="features")
    ## Load model
    model_lr = LinearRegressionModel.load("./model/linear_regression/lr_basic")
    model_rf = RandomForestRegressionModel.load("./model/random_forest/rf_basic")
    model_gbt = BTRegressionModel.load("./model/gradient_boosted/gbt_basic")


    main()


import streamlit as st
import numpy as np
import pandas as pd
from utils import _initialize_spark
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.functions import udf, col
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel, DecisionTreeRegressionModel, IsotonicRegressionModel, FMRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
import plotly.express as px


def tranformFetures(X, assembler):
    # Tạo bản sao để tránh ảnh hưởng dữ liệu gốc
    X_ = X.copy()
    ###########################




    ###########################
    st.write("tranform")
    return X_tranform

def prediction(samples, model):
    st.write("predict")
    # Encode dữ liệu
    X = tranformFetures(samples, assembler)
    # Predict
    return model.predict(X)

def LR_model():
    st.subheader('Mô hình Linear Regression')
    st.write('#### Sample dataset', pd_df)

    # Chọn dữ liệu từ mẫu
    selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
    selected_rows = pd_df.loc[selected_indices]
    st.write('#### Kết quả')

    if st.button('Dự đoán'):
        if not selected_rows.empty:
            X = selected_rows.iloc[:, :-1]
            pred = prediction(X, model_lr)

            # Xuất ra màn hình
            st.write("predict", pred)
            results = pd.DataFrame({'Giá dự đoán': pred,
                                        'Giá thực tế': selected_rows.TongGia})
            st.write(results)
        else:
            st.error('Hãy chọn dữ liệu trước')



def RF_model():
    st.subheader('Mô hình Random Forest')
    st.write('#### Sample dataset', pd_df)

    # Chọn dữ liệu từ mẫu
    selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
    selected_rows = pd_df.loc[selected_indices]
    st.write('#### Kết quả')

    if st.button('Dự đoán'):
        if not selected_rows.empty:
            X = selected_rows.iloc[:, :-1]
            pred = prediction(X, model_rf)

            # Xuất ra màn hình
            results = pd.DataFrame({'Giá dự đoán': pred,
                                        'Giá thực tế': selected_rows.TongGia})
            st.write(results)
        else:
            st.error('Hãy chọn dữ liệu trước')



def GBT_model():
    st.subheader('Mô hình Mô hình Gradient Boosting')
    st.write('#### Sample dataset', pd_df)

    # Chọn dữ liệu từ mẫu
    selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
    selected_rows = pd_df.loc[selected_indices]
    st.write('#### Kết quả')

    if st.button('Dự đoán'):
        if not selected_rows.empty:
            X = selected_rows.iloc[:, :-1]
            pred = prediction(X, model_gbt)

            # Xuất ra màn hình
            results = pd.DataFrame({'Giá dự đoán': pred,
                                    'Giá thực tế': selected_rows.TongGia})
            st.write(results)
        else:
            st.error('Hãy chọn dữ liệu trước')



def DT_model():
    st.subheader('Mô hình Decision Tree')
    st.write('#### Sample dataset', pd_df)

    # Chọn dữ liệu từ mẫu
    selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
    selected_rows = pd_df.loc[selected_indices]
    st.write('#### Kết quả')

    if st.button('Dự đoán'):
        if not selected_rows.empty:
            X = selected_rows.iloc[:, :-1]
            pred = prediction(X, model_dt)

            # Xuất ra màn hình
            results = pd.DataFrame({'Giá dự đoán': pred,
                                        'Giá thực tế': selected_rows.TongGia})
            st.write(results)
        else:
            st.error('Hãy chọn dữ liệu trước')



def IR_model():
    st.subheader('Mô hình Isotonic Regression')
    st.write('#### Sample dataset', pd_df)

    # Chọn dữ liệu từ mẫu
    selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
    selected_rows = pd_df.loc[selected_indices]
    st.write('#### Kết quả')

    if st.button('Dự đoán'):
        if not selected_rows.empty:
            X = selected_rows.iloc[:, :-1]
            pred = prediction(X, model_ir)

            # Xuất ra màn hình
            results = pd.DataFrame({'Giá dự đoán': pred,
                                    'Giá thực tế': selected_rows.TongGia})
            st.write(results)
        else:
            st.error('Hãy chọn dữ liệu trước')


def FMR_model():
    st.subheader('Mô hình FMR')
    st.write('#### Sample dataset', pd_df)

    # Chọn dữ liệu từ mẫu
    selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
    selected_rows = pd_df.loc[selected_indices]
    st.write('#### Kết quả')

    if st.button('Dự đoán'):
        if not selected_rows.empty:
            X = selected_rows.iloc[:, :-1]
            pred = prediction(X, model_ir)

            # Xuất ra màn hình
            results = pd.DataFrame({'Giá dự đoán': pred,
                                        'Giá thực tế': selected_rows.TongGia})
            st.write(results)
        else:
            st.error('Hãy chọn dữ liệu trước')

def creat_dashboard():
    st.subheader('Dashboard')
    fig = px.histogram(pd_df, x="Tinh", color="LoaiBDS")
        
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title('Dự đoán giá bất động sản')
    #st.plotly_chart(fig, use_container_width=True)
    #fig = ff.create_distplot(
        # hist_data, group_labels)
    #st.plotly_chart(fig, use_container_width=True)
    #st.line_chart(pd_df['LoaiBDS_idx','TongGia'])
    model_list = ['Dashboard',
                    'Mô hình Linear Regression',
                    'Mô hình Random Forest',
                    'Mô hình Gradient Boosting',
                    'Mô hình Decision Tree',
                    'Mô hình Isotonic Regression',
                    'Mô hình FMR']
    choice_model = st.sidebar.selectbox('Mô hình huấn luyện trên:', model_list)


    if choice_model =='Dashboard':
        creat_dashboard()
    elif choice_model == 'Mô hình Linear Regression':
        LR_model()

    elif choice_model == 'Mô hình Random Forest':
        RF_model()

    elif choice_model == 'Mô hình Gradient Boosting':
        GBT_model()

    elif choice_model == 'Mô hình Decision Tree':
        DT_model()

    elif choice_model == 'Mô hình Isotonic Regression':
        IR_model()

    elif choice_model == 'Mô hình FMR':
        FMR_model()

if __name__ == '__main__':
    spark, sc = _initialize_spark()
    ## Load dataset
    df = spark.read.format('org.apache.spark.sql.json').load("./data/clean/clean.json")
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
    data = data.fillna(0)
    pd_df = data.toPandas()
    features = data.columns
    features = [ele for ele in features if ele not in ['MaTin','TongGia','Gia/m2']]
    assembler = VectorAssembler(inputCols = features, outputCol="features")
    standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

    ## Load model
    model_lr = LinearRegressionModel.load("./model/linear_regression/lr_basic")
    model_rf = RandomForestRegressionModel.load("./model/random_forest/rf_basic")
    model_gbt = GBTRegressionModel.load("./model/gradient_boosted/gbt_basic")
    model_dt = DecisionTreeRegressionModel.load("./model/decision_tree/dt_basic")
    model_ir = IsotonicRegressionModel.load("./model/isotonic_regression/ir_basic")
    model_fmr = FMRegressionModel.load("./model/factorization_machines_regression/fmr_basic")


    main()


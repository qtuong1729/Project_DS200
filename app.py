import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep

from utils import _initialize_spark
from pyspark.sql.types import *
from pyspark.sql import functions as f
from pyspark.sql.functions import udf, col
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel, DecisionTreeRegressionModel, IsotonicRegressionModel, FMRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

from utils import *
from crawl_url import *
from crawl_data import *
from clean_data import *
from train_model import *



@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

@st.cache
def modelLoading():
    global model_lr_rmo, model_rf_rmo, model_gbt_rmo, model_dt_rmo, model_ir_rmo
    with st.spinner('Load model set ...'):
        model_lr_rmo = LinearRegressionModel.load("./model/linear_regression/lr_outlierRm")
        model_rf_rmo = RandomForestRegressionModel.load("./model/random_forest/rf_outlierRm")
        model_gbt_rmo = GBTRegressionModel.load("./model/gradient_boosted/gbt_outlierRm")
        model_dt_rmo = DecisionTreeRegressionModel.load("./model/decision_tree/dt_outlierRm")
        model_ir_rmo = IsotonicRegressionModel.load("./model/isotonic_regression/ir_outlierRm")

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

def load_sample_data():
    # Chọn dữ liệu từ mẫu
    selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
    selected_rows = pd_df.loc[selected_indices]
    st.write('#### Kết quả')

    if st.button('Dự đoán'):
        if not selected_rows.empty:
            X = selected_rows.iloc[:, :-1]
            pred = prediction(X, model)

            # Xuất ra màn hình
            st.write("predict", pred)
            results = pd.DataFrame({
                'Giá dự đoán': pred,
                'Giá thực tế': selected_rows.TongGia
                })
            st.write(results)
        else:
            st.error('Hãy chọn dữ liệu trước')

def inser_data():
    with st.form("Nhập dữ liệu"):
        loaiBDS = st.text_input("Loại BDS*")
        dienTich = st.text_input("Diện Tích*")
        tinh = st.text_input("Tỉnh\Thành phố*")
        hienTrangNha = st.text_input("Hiện Trạng Nhà")
        viTri = st.text_input("Vị trí")
        phongNgu = st.text_input("Số phòng ngủ")
        phongTam = st.text_input("Số phòng tắm")
        tang = st.text_input("Số tầng")
#        with st.expander("Thêm thông tin"):
#            loaiBDS = st.text_input("Loại BDS")
#            dienTich = st.text_input("Diện Tích")
#            hienTrangNha = st.text_input("Hiện Trạng Nhà")

        submitted = st.form_submit_button("Dự Đoán")

        if submitted:
            data_submitted = {'LoaiBDS' : loaiBDS,
                                'DienTich' : dienTich,
                                'Tinh': tinh,
                                'hienTrangNha': hienTrangNha,
                                'ViTri': viTri,
                                'PhongNgu': phongNgu,
                                'PhongTam': phongTam,
                                'Tang': tang}
            X = pd.DataFrame(data_submitted, index=[0])
            pred = prediction(X, model)

            # Xuất ra màn hình
            st.write("predict", pred)
            results = pd.DataFrame({'Giá dự đoán': pred,
                                        'Giá thực tế': selected_rows.TongGia})
            st.write(results)

def get_data_from_URL():
    st.write('#### Crawl URL')

def model_page(model_name, model):
    option_list = ['Dữ liệu mẫu', 'Nhập dữ liệu', 'Crawl dữ liệu từ URL']
    
    choice_input = st.sidebar.selectbox('Cách nhập dữ liệu', option_list)    
    st.subheader(model_name)
    if choice_input == 'Dữ liệu mẫu':
        st.write('#### Sample dataset', pd_df)
        load_sample_data()

    elif choice_input == 'Nhập dữ liệu':
        inser_data()

    elif choice_input == 'Crawl dữ liệu từ URL':
        get_data_from_URL()

def create_dashboard(df):
    st.subheader('Dashboard')

    col1, col2 = st.columns(2)
    col1.metric(label="Số lượng dự án", value=df.shape[0])
    col2.metric(label="Giá tiền trung bình mỗi dự án",
                value="{:,} VND".format(round(df['TongGia'].mean() * 1000)))

    fig1 = px.histogram(pd_df, x="Tinh", color="LoaiBDS", labels={
                     "Tinh": "Tỉnh(Thành phố)",
                     "LoaiBDS": "Loại BDS"
                 },)

    pd_date = pd_df.copy()
    pd_date['NgayDangBan'] = pd.to_datetime(pd_date['NgayDangBan']).dt.date
    
    fig_date = px.histogram(pd_date, x="NgayDangBan", labels={
                        "NgayDangBan": "Ngày Đăng bán",
                    },)
    
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig_date, use_container_width=True)

    fig_col2, fig_col3 = st.columns(2)

    fig2 = px.histogram(pd_df, x="LoaiBDS", y="TongGia", histfunc='avg', labels = {
            "LoaiBDS": "Loại BDS",
            "TongGia": "price"
        })

    pd_df2 = df.groupby('LoaiBDS').size().reset_index(name='Observation')
    fig3 = px.pie(pd_df2, values='Observation', names='LoaiBDS', title = 'Tỷ lệ các loại BDS')

    fig_col2.plotly_chart(fig2)
    fig_col3.plotly_chart(fig3)

    st.write(pd_df)

def main():
    st.title('Dự đoán giá bất động sản')
    model_list = ['Dashboard',
                    'Mô hình Linear Regression',
                    'Mô hình Random Forest',
                    'Mô hình Gradient Boosting',
                    'Mô hình Decision Tree',
                    'Mô hình Isotonic Regression']
    global choice_model
    choice_model = st.sidebar.selectbox('Mô hình huấn luyện trên:', model_list)


    if choice_model =='Dashboard':
        create_dashboard(pd_df)
    elif choice_model == 'Mô hình Linear Regression':
        model_page(choice_model, model_lr_rmo)

    elif choice_model == 'Mô hình Random Forest':
        model_page(choice_model, model_rf_rmo)

    elif choice_model == 'Mô hình Gradient Boosting':
        model_page(choice_model, model_gbt_rmo)

    elif choice_model == 'Mô hình Decision Tree':
        model_page(choice_model, model_dt_rmo)

    elif choice_model == 'Mô hình Isotonic Regression':
        model_page(choice_model, model_ir_rmo)


if __name__ == '__main__':
    spark, sc = _initialize_spark()
    st.set_page_config(layout="wide")
    ## Load dataset
    with st.spinner('Load data...'):
        df = spark.read.format('org.apache.spark.sql.json').load("./data/clean/clean.json")
    data = df.drop(*['id', 'MoTa'])
    #st.write("data ready")
    data = data.fillna(0)
    pd_df = data.toPandas()

    ## Load model
    model_lr_rmo, model_rf_rmo, model_gbt_rmo, model_dt_rmo, model_ir_rmo = \
    (lambda n: [None for _ in range(n)])(5)

    modelLoading()
    output = st.empty()
        
    main()

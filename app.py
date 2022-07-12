from pyspark.rdd import RDD
from pyspark.sql import Row
import streamlit as st
from utils import _initialize_spark

st.write("# :tada: Hello Pyspark")
st.write("[Link to Spark window](http://localhost:4040)")

st.write("## Create RDD from a Python list")

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


# spark.stop()
if __name__ == '__main__':
    spark, sc = _initialize_spark()
    st.write("create sc")
    df = spark.read.format('org.apache.spark.sql.json').load("qtuong1729/Project_DS200/clean/clean.json")
    st.write("data ready")
    df=df.withColumnRenamed("P. sinh hoạt chung","Phòng sinh hoạt chung")
    df=df.withColumnRenamed("T.T thương mại tòa nhà","TT thương mại tòa nhà")
    df=df.withColumnRenamed("T.T thương mại","TT thương mại")
    df=df.withColumnRenamed("Bàn ghế P.Khách","Bàn ghế PKhách")
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
        
    for col_name in cols:
        df = df.withColumn(col_name, col(col_name).cast('int'))
    data = df.drop(*['TienIchGanDat','id','NgayDangBan', 'MoTa_Vec'])
    data.haed(10)
    st.write("data ready")
    ## Load dataset
    ## Load model


    main()
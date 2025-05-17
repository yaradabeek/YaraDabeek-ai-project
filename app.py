import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.io as pio
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# بيانات وهمية تم جمعها من اللعبة
data = [
    {"date": "5/9/2025, 11:53:50 PM", "duration": "11 seconds", "jumps": 5, "score": 0, "obstaclesPassed": 0},
    {"date": "5/9/2025, 11:54:16 PM", "duration": "14 seconds", "jumps": 10, "score": 1, "obstaclesPassed": 1},
    {"date": "5/9/2025, 11:55:19 PM", "duration": "67 seconds", "jumps": 40, "score": 10, "obstaclesPassed": 12},
    {"date": "5/9/2025, 11:56:46 PM", "duration": "153 seconds", "jumps": 100, "score": 30, "obstaclesPassed": 35},
    {"date": "5/10/2025, 1:12:26 PM", "duration": "20 seconds", "jumps": 8, "score": 1, "obstaclesPassed": 2},
    {"date": "5/11/2025, 9:12:22 PM", "duration": "20 seconds", "jumps": 9, "score": 2, "obstaclesPassed": 3},
    {"date": "5/11/2025, 9:13:01 PM", "duration": "59 seconds", "jumps": 55, "score": 12, "obstaclesPassed": 14},
    {"date": "5/11/2025, 9:15:31 PM", "duration": "209 seconds", "jumps": 120, "score": 40, "obstaclesPassed": 42},
    {"date": "5/11/2025, 9:16:00 PM", "duration": "180 seconds", "jumps": 85, "score": 28, "obstaclesPassed": 30},
    {"date": "5/11/2025, 9:20:00 PM", "duration": "90 seconds", "jumps": 50, "score": 15, "obstaclesPassed": 17}
]

df = pd.DataFrame(data)

# تحويل المدة إلى ثواني
df['duration_seconds'] = df['duration'].apply(lambda x: int(x.split(' ')[0]))

# تطبيق K-means clustering
X = df[['jumps', 'duration_seconds', 'score']]
n_clusters = st.slider('اختار عدد الكتل في K-means:', 2, 10, 3)
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
df['cluster'] = kmeans.labels_

# تطبيق DecisionTreeClassifier للتنبؤ بالفشل
df['failure'] = df['score'].apply(lambda x: 1 if x < 10 else 0)
X = df[['jumps', 'duration_seconds', 'score']]
y = df['failure']
clf = DecisionTreeClassifier()
clf.fit(X, y)
df['predicted_failure'] = clf.predict(X)

# إضافة تصنيف اللاعبين بناءً على النقاط
def classify_player(row):
    if row['score'] > 30:
        return 'محترف'
    elif 10 <= row['score'] <= 30:
        return 'متوسط'
    else:
        return 'مبتدئ'

# إضافة التصنيف إلى DataFrame
df['player_classification'] = df.apply(classify_player, axis=1)

# عنوان التطبيق
st.title('تحليل بيانات اللعبة باستخدام التعلم الآلي')

# عرض الجدول الأول (بيانات اللعبة)
st.write("### بيانات اللعبة")
st.dataframe(df[['date', 'duration', 'jumps', 'score', 'obstaclesPassed', 'duration_seconds']])

# عرض الجدول الثاني (تصنيف اللاعبين)
st.write("### جدول تصنيف اللاعبين")
st.dataframe(df[['date', 'score', 'player_classification']])

# رسم بياني لـ K-means Clustering
st.write("### رسم بياني لـ K-means Clustering")
fig, ax = plt.subplots()
sns.scatterplot(x='jumps', y='score', hue='cluster', data=df, palette='Set2', ax=ax)
st.pyplot(fig)

# رسم بياني للتنبؤ بالفشل
st.write("### رسم بياني للتنبؤ بالفشل")
fig, ax = plt.subplots()
sns.scatterplot(x='jumps', y='score', hue='predicted_failure', data=df, palette='coolwarm', ax=ax)
st.pyplot(fig)

# إنشاء نموذج الانحدار الخطي للتنبؤ بالنقاط
reg = LinearRegression()
X = df[['jumps', 'duration_seconds']]
y = df['score']
reg.fit(X, y)
df['predicted_score'] = reg.predict(X)

# عرض التنبؤ بالنقاط بناءً على عدد القفزات والمدة
st.write("### التنبؤ بالنقاط بناءً على عدد القفزات والمدة:")
st.dataframe(df[['jumps', 'duration_seconds', 'predicted_score']])

# رسم بياني تفاعلي باستخدام Plotly
fig = px.scatter(df, x='jumps', y='score', color='cluster', title="توزيع اللاعبين حسب التصنيف")
st.plotly_chart(fig)

# زر لتحميل البيانات كـ CSV
st.download_button(
    label="تحميل البيانات كـ CSV",
    data=df.to_csv(index=False),  # تحويل البيانات إلى CSV
    file_name="player_data.csv",
    mime="text/csv"
)

# وظيفة لتحويل البيانات إلى PDF
def create_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)
    
    # إضافة البيانات إلى PDF
    for index, row in df.iterrows():
        text.textLine(f"{row['date']} | {row['score']} | {row['player_classification']}")
    
    c.drawText(text)
    c.showPage()
    c.save()
    
    # إعادة التحميل
    buffer.seek(0)
    return buffer

# تحميل الملف كـ PDF
st.download_button(
    label="تحميل البيانات كـ PDF",
    data=create_pdf(),
    file_name="player_data.pdf",
    mime="application/pdf"
)

# حفظ الرسم البياني كصورة PNG
pio.write_image(fig, 'chart.png')

# عرض زر لتحميل الصورة
with open("chart.png", "rb") as file:
    st.download_button(
        label="تحميل الرسم البياني كصورة",
        data=file,
        file_name="chart.png",
        mime="image/png"
    )

# -*- coding: utf-8 -*-




from kivy.uix.image import Image
from kivy.app import App
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from pickle import load
import tensorflow as tf
from keras.models import model_from_json

global sc
global encoder
global interpreter
global loaded_model
global df
global live
global fakedata
global SMA


sc = load(open(r"C:\Users\Ayori\COM 724\assessment\crypto\scale2.pkl", 'rb'))
# encoder= load(open('le.pkl', 'rb'))
file = open(r"C:\Users\Ayori\COM 724\assessment\crypto\lelast.obj",'rb')
encoder = load(file)
file.close()

model_path = r"C:\Users\Ayori\COM 724\assessment\crypto\neuralregressor_small4.tflite"
interpreter = tf.lite.Interpreter(model_path)
print("Model Loaded Successfully.")
df= pd.read_csv(r"C:\Users\Ayori\COM 724\assessment\crypto\data3.csv")



json_file = open(r"C:\Users\Ayori\COM 724\assessment\crypto\m_finallast__2.JSON", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"C:\Users\Ayori\COM 724\assessment\crypto\m_finallast__2.h5")
print("Loaded model from disk")

def SMA(arr):
    
    window_size = 6
    
    i = 0
    moving_averages = []
    while i < len(arr) - window_size + 1:

        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i : i + window_size]

        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
    
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
    
        # Shift window to right by one position
        i += 1
    return (moving_averages,[i for i in range(len(moving_averages))])


def fakedata(x, DF=df):
    DF= DF.to_numpy()
    # df=df.T[:-1].
    # in__=
    # print(in__.shape)
    # #     in__=sc.transform(in__)T
    unique,counts=np.unique(DF.T[0],return_counts=True)
    counts1=np.cumsum(counts)
    counts1=counts1.tolist()
    #print(counts1)
    indx=[[i,j] for j,i in zip(counts1,[0]+counts1[:-1])]
    indx_dict={i:j for i,j in zip(unique,indx)}
    # print(indx_dict)
    a=indx_dict[int(x)][0]
    b=indx_dict[int(x)][1]
    length=b-a
    result=DF[a:b]
    # for i in range(int(10000/length)):
    #     result=np.vstack((result,df[a:b]))
    return(result)



class SOLiGence(App):
    def build(self):
        self.major_layout=BoxLayout(orientation='vertical')
        self.main_layout=BoxLayout()
        self.font='40sp'
        self.speed=20
        self.length=500
        # with self.main_layout.canvas:
        #     Color(0, 1, 0, 0.25)
        #     Rectangle(pos=self.main_layout.pos, size=self.main_layout.size)
        self.firstlayout=BoxLayout(orientation='vertical',  size_hint=(.5,1),size=(Window.width, Window.height*1.5))
        self.mid_layout=BoxLayout(orientation='vertical',  size_hint=(1,1))
        self.right_layout=BoxLayout(orientation='vertical', size_hint=(0.2,0.3))
        self.scroll=ScrollView(do_scroll=True,size_hint=(.3, None), size=(Window.width, Window.height))
        
        self.coins_ = BoxLayout(orientation="horizontal", size_hint=(1,0.05))
        self.c1 = Label(text="Base Token",size_hint=(.1,1))
        self.Cinput10=TextInput(text='ADA', size_hint=(.9,1))
        self.c1_ = Button(size_hint=(.1,1))
        self.coins_.add_widget(self.c1)
        self.coins_.add_widget(self.Cinput10)
        self.coins_.add_widget(self.c1_)
        self.major_layout.add_widget(self.coins_)
        
        self.coins_container = BoxLayout(orientation="horizontal", size_hint=(1,0.05))
        self.all_coins=['ADA', 'ATOM','BAKE', 'BCD','CELO',
                        'CFX', 'CHSB', 'CRV', 'DENT',  
                        'LRC', 'DOGE', 'XRP','UNI', 'DOT','WBTC',
                        'ENJ','IOST','LRC','NANO','ONE','PAX',
                        'QTUM', 'REN','USDC', 'USDT', 'VET','XLM']
        self.t1 = Button(text=self.all_coins[0],size_hint=(.07,1))
        self.t2 = Button(text=self.all_coins[1],size_hint=(.07,1))
        self.t3 = Button(text=self.all_coins[2],size_hint=(.07,1))
        self.t4 = Button(text=self.all_coins[3],size_hint=(.07,1))
        self.t5 = Button(text=self.all_coins[4],size_hint=(.07,1))
        self.t6 = Button(text=self.all_coins[5],size_hint=(.07,1))
        self.t7 = Button(text=self.all_coins[6],size_hint=(.07,1))
        self.t8 = Button(text=self.all_coins[7],size_hint=(.07,1))
        self.t9 = Button(text=self.all_coins[8],size_hint=(.07,1))
        self.t10 = Button(text=self.all_coins[9],size_hint=(.07,1))
        self.t11 = Button(text=self.all_coins[10],size_hint=(.07,1))
        self.t12 = Button(text=self.all_coins[11],size_hint=(.07,1))
        self.t13 = Button(text=self.all_coins[12],size_hint=(.07,1))
        self.coins_container.add_widget(self.t1)
        self.coins_container.add_widget(self.t2)
        self.coins_container.add_widget(self.t3)
        self.coins_container.add_widget(self.t4)
        self.coins_container.add_widget(self.t5)
        self.coins_container.add_widget(self.t6)
        self.coins_container.add_widget(self.t7)
        self.coins_container.add_widget(self.t8)
        self.coins_container.add_widget(self.t9)
        self.coins_container.add_widget(self.t10)
        self.coins_container.add_widget(self.t11)
        self.coins_container.add_widget(self.t12)
        self.coins_container.add_widget(self.t13)
        self.major_layout.add_widget(self.coins_container)
        
        
        # self.coins_compare = BoxLayout(orientation="horizontal", size_hint=(1,0.05))
        # self.var10 = Label(text="Token to \ncompare with",size_hint=(.1,1))
        # self.input10=TextInput(size_hint=(.9,1))
        # self.coins_compare.add_widget(self.var10)
        # self.coins_compare.add_widget(self.input10)
        # self.major_layout.add_widget(self.coins_compare)
        
        
        self.coins_container1 = BoxLayout(orientation="horizontal", size_hint=(1,0.05))
        self.t11 = Button(text=self.all_coins[13],size_hint=(.07,1))
        self.t21 = Button(text=self.all_coins[14],size_hint=(.07,1))
        self.t31 = Button(text=self.all_coins[15],size_hint=(.07,1))
        self.t41 = Button(text=self.all_coins[16],size_hint=(.07,1))
        self.t51 = Button(text=self.all_coins[17],size_hint=(.07,1))
        self.t61 = Button(text=self.all_coins[18],size_hint=(.07,1))
        self.t71 = Button(text=self.all_coins[19],size_hint=(.07,1))
        self.t81 = Button(text=self.all_coins[20],size_hint=(.07,1))
        self.t91 = Button(text=self.all_coins[21],size_hint=(.07,1))
        self.t101 = Button(text=self.all_coins[22],size_hint=(.07,1))
        self.t111 = Button(text=self.all_coins[23],size_hint=(.07,1))
        self.t121 = Button(text=self.all_coins[24],size_hint=(.07,1))
        self.t131 = Button(text=self.all_coins[25],size_hint=(.07,1))
        self.coins_container1.add_widget(self.t11)
        self.coins_container1.add_widget(self.t21)
        self.coins_container1.add_widget(self.t31)
        self.coins_container1.add_widget(self.t41)
        self.coins_container1.add_widget(self.t51)
        self.coins_container1.add_widget(self.t61)
        self.coins_container1.add_widget(self.t71)
        self.coins_container1.add_widget(self.t81)
        self.coins_container1.add_widget(self.t91)
        self.coins_container1.add_widget(self.t101)
        self.coins_container1.add_widget(self.t111)
        self.coins_container1.add_widget(self.t121)
        self.coins_container1.add_widget(self.t131)
        self.major_layout.add_widget(self.coins_container1)
        
        self.t1.bind(on_press=self.symbol)
        self.t2.bind(on_press=self.symbol)
        self.t3.bind(on_press=self.symbol)
        self.t4.bind(on_press=self.symbol)
        self.t5.bind(on_press=self.symbol)
        self.t6.bind(on_press=self.symbol)
        self.t7.bind(on_press=self.symbol)
        self.t8.bind(on_press=self.symbol)
        self.t9.bind(on_press=self.symbol)
        self.t10.bind(on_press=self.symbol)
        self.t11.bind(on_press=self.symbol)
        self.t12.bind(on_press=self.symbol)
        self.t13.bind(on_press=self.symbol)
        self.t11.bind(on_press=self.symbol)
        self.t21.bind(on_press=self.symbol)
        self.t31.bind(on_press=self.symbol)
        self.t41.bind(on_press=self.symbol)
        self.t51.bind(on_press=self.symbol)
        self.t61.bind(on_press=self.symbol)
        self.t71.bind(on_press=self.symbol)
        self.t81.bind(on_press=self.symbol)
        self.t91.bind(on_press=self.symbol)
        self.t101.bind(on_press=self.symbol)
        self.t111.bind(on_press=self.symbol)
        self.t121.bind(on_press=self.symbol)
        self.t131.bind(on_press=self.symbol)
        
        
        
        
        
        self.one=BoxLayout(orientation="horizontal")
        self.var1=Label(text="Token \nSymbol",size_hint=(.3,1))
        self.input1=TextInput(text='ADA', size_hint=(.7,1),multiline=False,font_size=self.font)
        self.input1.bind(on_text_validate=self.symbol)
        self.one.add_widget(self.var1)
        self.one.add_widget(self.input1)
        
        self.two=BoxLayout(orientation="horizontal")
        self.var2=Label(text="Days \nforward",size_hint=(.3,1))
        self.input2=TextInput(text='0',size_hint=(.7,1),multiline=False,font_size=self.font)
        self.input2.bind(on_text_validate=self.on_days)
        self.two.add_widget(self.var2)
        self.two.add_widget(self.input2)
        
        self.three=BoxLayout(orientation="horizontal")
        self.var3=Label(text="price",size_hint=(.3,1))
        self.input3=TextInput(text='current',size_hint=(.7,1),multiline=False,font_size=self.font)
        self.input3.bind(on_text_validate=self.on_price)
        self.three.add_widget(self.var3)
        self.three.add_widget(self.input3)
        
        self.four=BoxLayout(orientation="horizontal")
        self.var4=Label(text="Speed",size_hint=(.3,1))
        self.input4=TextInput(text=str(self.speed),size_hint=(.7,1),multiline=False,font_size=self.font)
        self.input4.bind(on_text_validate=self.on_enter_speed)
        self.four.add_widget(self.var4)
        self.four.add_widget(self.input4)
        
        self.five=BoxLayout(orientation="horizontal")
        self.var5=Label(text="Length",size_hint=(.3,1))
        self.input5=TextInput(text=str(self.length),size_hint=(.7,1),multiline=False,font_size=self.font)
        self.input5.bind(on_text_validate=self.on_enter_length)
        self.five.add_widget(self.var5)
        self.five.add_widget(self.input5)
        
        self.six=BoxLayout(orientation="horizontal")
        self.var6=Label(text="Symbol",size_hint=(.3,1))
        self.input6=TextInput(size_hint=(.7,1))
        self.six.add_widget(self.var6)
        self.six.add_widget(self.input6)
        
        
        self.seven=BoxLayout(orientation="horizontal")
        self.var7=Label(text="Symbol",size_hint=(.3,1))
        self.input7=TextInput(size_hint=(.7,1))
        self.seven.add_widget(self.var7)
        self.seven.add_widget(self.input7)
        
        
        self.eight=BoxLayout(orientation="horizontal")
        self.var8=Label(text="Symbol",size_hint=(.3,1))
        self.input8=TextInput(size_hint=(.7,1))
        self.eight.add_widget(self.var8)
        self.eight.add_widget(self.input8)
        
        
        self.nine=BoxLayout(orientation="horizontal")
        self.var9=Label(text="Symbol",size_hint=(.3,1))
        self.input9=TextInput(size_hint=(.7,1))
        self.nine.add_widget(self.var9)
        self.nine.add_widget(self.input9)
        
        
        
        self.ten=BoxLayout(orientation="horizontal")
        self.var10=Label(text="Symbol",size_hint=(.3,1))
        self.input10=TextInput(size_hint=(.7,1))
        self.ten.add_widget(self.var10)
        self.ten.add_widget(self.input10)
        
        
        
        self.eleven=BoxLayout(orientation="horizontal")
        self.var11=Label(text="Symbol",size_hint=(.3,1))
        self.input11=TextInput(size_hint=(.7,1))
        self.eleven.add_widget(self.var11)
        self.eleven.add_widget(self.input11)
        
        
        self.twelve=BoxLayout(orientation="horizontal")
        self.var12=Label(text="Symbol",size_hint=(.3,1))
        self.input12=TextInput(size_hint=(.7,1))
        self.twelve.add_widget(self.var12)
        self.twelve.add_widget(self.input12)
        
        
        self.thirteen=BoxLayout(orientation="horizontal")
        self.var13=Label(text="Symbol",size_hint=(.3,1))
        self.input13=TextInput(size_hint=(.7,1))
        self.thirteen.add_widget(self.var13)
        self.thirteen.add_widget(self.input13)
        
        
        self.firstlayout.add_widget(self.one)
        self.firstlayout.add_widget(self.two)
        self.firstlayout.add_widget(self.three)
        self.firstlayout.add_widget(self.four)
        self.firstlayout.add_widget(self.five)
        # self.firstlayout.add_widget(self.six)
        # self.firstlayout.add_widget(self.seven)
        # self.firstlayout.add_widget(self.eight)
        # self.firstlayout.add_widget(self.nine)
        # self.firstlayout.add_widget(self.ten)
        # self.firstlayout.add_widget(self.eleven)
        # self.firstlayout.add_widget(self.twelve)
        # self.firstlayout.add_widget(self.thirteen)
        # self.scroll.add_widget(self.firstlayout)
        
        
        
        self.filepath=r"C:\Users\Ayori\COM 724\assessment\crypto\pexels-monicore-134054.jpg"
        self.img = Image(source=self.filepath,
                    size_hint=(1, 1), allow_stretch=True, keep_ratio=False
                    )
        self.img_ = Image(source=self.filepath,
                    size_hint=(1, 1), allow_stretch=True, keep_ratio=False
                    )
        self.mid_layout2=BoxLayout(size_hint=(1,0.3))
        self.mid_layout_2=BoxLayout(size_hint=(1,.7))
        self.mid_layout_2.add_widget(self.img)
        self.mid_layout_2.add_widget(self.img_)
        self.img1 = Image(source=self.filepath,
                    size_hint=(0.5, 1), allow_stretch=True, keep_ratio=False
                    )
        # self.img2 = Image(source=self.filepath,
        #             size_hint=(0.2, 1), allow_stretch=True, keep_ratio=False
        #             )
        # self.img3 = Image(source=self.filepath,
        #             size_hint=(0.2, 1), allow_stretch=True, keep_ratio=False
        #             )
        # self.img4 = Image(source=self.filepath,
        #             size_hint=(.5, 1), allow_stretch=True, keep_ratio=False
        #             )
        self.pred_label=Label(text='Next x hours prediction',size_hint=(0.5, 1))
        self.imgT=cv2.imread(self.filepath,0)
        # self.img.bind(on_touch_move= self.update)
        self.d_buttons=BoxLayout(padding=5, size_hint=(1,0.1))
        self.left_d = Button(text='prev')
        self.right_d = Button(text='next')
        self.right_d.bind(on_press=self.next_graph)
        self.left_d.bind(on_press=self.prev_graph)
        self.d_buttons.add_widget(self.left_d)
        self.d_buttons.add_widget(self.right_d)
        self.d_label=Label(text='Foreign Markets are not worthy of local markets- NYTimes', size_hint=(1,0.2))
                
        self.mid_layout.add_widget(self.mid_layout_2)
        self.mid_layout2.add_widget(self.img1)
        self.mid_layout2.add_widget(self.pred_label)

        self.mid_layout.add_widget(self.mid_layout2)
        self.mid_layout.add_widget(self.d_buttons)
        self.mid_layout.add_widget(self.d_label)
        
        self.main_layout.add_widget(self.firstlayout)
        self.main_layout.add_widget(self.mid_layout)
        self.major_layout.add_widget(self.main_layout)
        
        
       
        self.coin_track=0
        self.current_coin=self.corr_coin=self.all_coins[self.coin_track]
        token=encoder.transform(np.array([self.current_coin]))
        token=token[0]
        fk=fakedata(token,df).T
        self.y=fk[7]
        self.x=[i for i in range(int(self.y.shape[0]))]
        self.y2=fk[7]
        self.x2=[i for i in range(int(self.y.shape[0]))]
        self.pred=self.model_predict(fk.T)
        self.news=['Crypto Helps Ukraine Defend Itself Against Russia\'s Invasion',
              'MetaGoblin NFTs by MetaBlaze: Income Generating Utility in BlazedApp and Perpetual Royalty Earnings, NFT Presale May 8th',
              'Nvidia Pays USD 5.5M Fine Over Crypto-Mining Disclosures',
              'Major Bitcoin & Crypto Companies Warn of \'Extreme\' Risk in Proof-of-Stake Systems',
              'Ethereum Needs to Pass These Three Tests Before Migrating to PoS']
        self.news_track=0
        self.track=0
        
        Clock.schedule_interval(self.animate_, 40/ 60.0)
        Clock.schedule_interval(self.news__, 8)
        return self.major_layout
    
    def on_enter_speed(self, value):
        txt=value.text
        self.speed=int(txt)
    def symbol(self,instance):
        try:
            txt=instance.text
            x=encoder.transform(np.array([txt]))
            # print(x)
            x=x[0]
            self.current_coin=txt
            fk=fakedata(x).T
            self.y=fk[7]
            self.x=[i for i in range(int(self.y.shape[0]))]
            self.pred=self.model_predict(fk.T)
            self.track=0
            self.Cinput10.text=self.current_coin
        except:
            self.input1.text='invalid'
            pass

        
    def on_enter_length(self, value):
        txt=value.text
        self.length=int(txt)
    
    def on_price(self,value):
        try:
            txt=float(value.text)
            # x=encoder.transform(np.array([txt]))
            # print(x)
            self.current_price=txt
            token=encoder.transform(np.array([self.current_coin]))
            token=token[0]
            df_=df.copy(deep=True)
            df_['price']=(df_['price']*0)+txt
            fk=fakedata(token,df_).T
            # self.y=fk[7]
            # self.x=[i for i in range(int(self.y.shape[0]))]
            self.pred=self.model_predict(fk.T)
            self.pred_label.text='predicted price with the given values\n '+str(self.pred[0])
            # self.track=0
            self.news__
        except:
            self.pred_label.text='invalid'
            pass
    
    # def on_price(self,value):
        
    #     txt=float(value.text)
    #     # x=encoder.transform(np.array([txt]))
    #     # print(x)
    #     self.current_price=txt
    #     df['price']=(df['price']*0)+txt
    #     token=encoder.transform(np.array([self.current_coin]))
    #     token=token[0]
    #     fk=fakedata(token,df).T
    #     # self.y=fk[7]
    #     # self.x=[i for i in range(int(self.y.shape[0]))]
    #     self.pred=self.model_predict(fk.T)
    #     # self.track=0
        
    def on_days(self,value):
        try:
            txt=int(value.text)
            # x=encoder.transform(np.array([txt]))
            # print(x)
            df['last_updated']=(df['last_updated']*0)+txt
            # x=x[0]
            self.current_day=txt
            token=encoder.transform(np.array([self.current_coin]))
            token=token[0]
            fk=fakedata(token,df).T
            # self.y=fk[7]
            # self.x=[i for i in range(int(self.y.shape[0]))]
            self.pred=self.model_predict(fk.T)
            _=self.news__
            print(self.pred)
            # self.track=0
            self.pred_label.text='using the given values, predicted price:\n one'+ str(self.current_coin)+'='+str(self.pred[0])
        except:
            self.pred_label.text='invalid'
            pass


    def next_graph(self,rect1):
        self.coin_track+=1
        print(self.coin_track)
        self.corr_coin=self.all_coins[self.coin_track]
        token=encoder.transform(np.array([self.corr_coin]))
        token=token[0]
        print(token)
        fk=fakedata(token,df).T
        self.y2=fk[7]
        self.x2=[i for i in range(int(self.y2.shape[0]))]
                
        
    
    
    
    def prev_graph(self,rect1):
        self.coin_track-=1
        self.corr_coin=self.all_coins[self.coin_track]
        token=encoder.transform(np.array([self.corr_coin]))
        token=token[0]
        fk=fakedata(token,df).T
        self.y2=fk[7]
        self.x2=[i for i in range(int(self.y2.shape[0]))]
        
        pass

    def news__(self,dt):
        if self.news_track>4:
            self.news_track=0
        self.d_label.text=self.news[self.news_track]
        
        self.news_track+=1
    # def on_enter_symbol(self, value):
    #     try:
    #         txt=value.text
    #         x=encoder.transform(np.array([txt]))
            
    #         # print(x)
    #         x=x[0]
    #         self.current_coin=x
    #         fk=fakedata(x).T
    #         self.y=fk[7]
    #         self.x=[i for i in range(int(self.y.shape[0]))]
            
    #         self.pred=self.model_predict(fk.T)
    #         self.track=0
    #     except:
    #         self.input1.text='invalid'
    #         pass
    #     # self.speed=int(txt)
    # def on_enter_days(self, value):
    #     try:
    #         txt=int(value.text)
    #         x=encoder.transform(np.array([txt]))
    #         # print(x)
    #         x=x[0]
    #         fk=fakedata(x).T
    #         self.y=fk[7]
    #         self.x=[i for i in range(int(self.y.shape[0]))]
            
    #         self.pred=self.model_predict(fk.T)
    #         self.track=0
    #     except:
    #         self.input1.text='invalid'
    #         pass
    #     txt=value.text
    #     self.speed=int(txt)
    # def on_enter_price(self, value):
    #     txt=value.text
    #     self.speed=int(txt)
    # def preprocess(self,x):
    #     return None
    #     pass
    #     # x=x[]
    def model_predict(self,x):
        in__=sc.transform(x)
        y=loaded_model.predict(in__)
        y=[i for i in y.T[0]]
        return y
    
    # def model_predict(self,x):
    #     in__=x.reshape(-1,17)
    #     # print(in__.shape)
    #     in__=sc.transform(in__)
    #     in__=in__.astype('float32')
    #     # print(in__.shape)
    #     input_details = interpreter.get_input_details()
    #     output_details = interpreter.get_output_details()
    #     interpreter.allocate_tensors()
    #     interpreter.set_tensor(input_details[0]['index'], in__)
        
       
    #     interpreter.invoke()
        
    #     # output_details[0]['index'] = the index which provides the input
    #     output_data = interpreter.get_tensor(output_details[0]['index'])
    #     return(int(output_data.flatten()[0])) 

    def animate_(self,dt):
        if self.track+self.length>len(self.x):
            self.track=0
        # plt.plot(self.x[self.track:self.track + self.length],self.y[self.track:self.track + self.length])
        # plt.xlim((self.x[self.track],self.x[self.track + self.length]))
        # plt.savefig("temp.png")
            
        self.update_window(self.x,self.y,self.pred,self.img)
        try:
            self.update_window(self.x2,self.y2,self.pred,self.img_,self.corr_coin) 
            self.update_bar(self.x2,self.y2,self.pred,self.img1)
        except:
            pass
        self.track+=self.speed
        pass
    def update_bar(self,x,y,pred,winname):
        self.fig2 = plt.Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.bar("A",y[self.track]-int(y[self.track]))
        self.ax2.bar("B",pred[self.track]-int(y[self.track]))
        self.ax2.set_ylim((0,y[self.track:self.track + self.length].max()))
        self.ax2.set_ylabel('current_price')
        self.ax2.set_xlabel('Time(s)')
        self.ax.grid(visible=True, which='both', axis='both')
        self.fig2.savefig("temp.png")
        imgT=cv2.imread("temp.png")
        os.remove("temp.png")
        frame = imgT
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        winname.texture = texture1
    def update_window(self,x,y,pred,winame,txt=False):
        if txt==False:
            txt=self.current_coin
            title_text='Hourly plot of the ' + txt + ' against price'
        else:
            mrk=min([self.y.shape[0],self.y2.shape[0]]) -1 
            corr=str(100* np.corrcoef(self.y[:mrk],self.y2[:mrk])[0][1])
            txt=self.corr_coin
            title_text='Hourly plot of the ' + txt + ' against price\n Correlation with base token = '+corr
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)
        
        
        self.ax.plot(x[self.track:self.track + self.length], y[self.track:self.track + self.length])
        a,b=SMA(y[self.track:self.track + self.length])        
        self.ax.plot([i + self.track for i in b],a) 
        # self.ax.plot(x[self.track:self.track + self.length], pred[self.track:self.track + self.length])
        self.ax.set_ylabel('current_price')
        self.ax.set_xlabel('Time(s)')
        self.ax.grid(visible=True, which='both', axis='both')
        self.ax.set_title(title_text)
        self.fig.savefig("temp.png")
        imgT=cv2.imread("temp.png")
        os.remove("temp.png")
        frame = imgT
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        winame.texture = texture1
    # def update_second_window(self,x,y,pred):
        
    def update_token(self, graph):
        pass
    def update_day(self, graph):
        pass
    def save_image(self,rect1):
        pass
        
    def clusterframe(self,rect1):
        
        pass
    
        
    
    def update(self, w,p):
        pass
    
    
    
    
    
    
    
    
    
    def create__mask(self, rect1):
        
        pass
        
    
    
 
if __name__ == '__main__':
    SOLiGence().run()
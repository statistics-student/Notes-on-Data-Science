import pandas as pd
import requests
from urllib.parse import urlencode

base_url="https://dc.qiumibao.com/shuju/public/index.php?"
headers={
    'Referer':'https://data.zhibo8.cc/pc_main_data/',
    'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/\
    537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36'    
}
players=['进球','助攻','黄牌','红牌','两黄变红','射门','射正','乌龙球','禁区内进球',
         '禁区外进球','左脚进球','右脚进球','点球进球','头球进球','定位球破门','失球',
         '越位','抢断','拦截','回球','解围','抢断成功','抢断失败','犯规','被犯规','单挑成功',
         '单挑失败','争顶成功','争顶失败','出场时间','传球','传中','角球','短传','长传','直塞'
         ,'扑出点球','扑出射门','扑救']
teams=['进球','总失球数','助攻','黄牌','红牌','两黄变红','射门','射正','乌龙球','禁区内进球',
       '禁区外进球','左脚进球','右脚进球','点球被罚进','头球得分','定位球破门','越位','抢断',
       '拦截','回球','解围','抢断成功','抢断失败','犯规','被犯规','单挑成功','单挑失败',
       '头球争顶成功率','争顶失败','传球','传中','角球','短传','长传','直塞','扑出点球','扑救']
transforms=['冬窗','夏窗','巴塞罗那','马德里竞技','塞维利亚','皇家马德里','阿拉维斯','贝蒂斯',
            '赫塔费','瓦伦西亚','赫罗纳','莱万特','维戈塞尔塔','巴拉多利德','埃瓦尔','西班牙人',
            '皇家社会','莱加内斯','毕尔巴鄂','比利亚雷亚尔','巴列卡诺','韦斯卡']
tabs={'赛程':[''],'积分榜':[''],'球员榜':players,'球队榜':teams,'转会':transforms}
def get_data(tab,Type):
    params={
        '_url':'/data/index',
        'league': '西甲',
        'tab': tab,
        'type': Type,
        'year': '[year]'}
    if Type!='':
        pass
    else:
        params.pop('type')
    url=base_url+urlencode(params)
    try:
        response=requests.get(url,headers=headers)
        if response.status_code==200:
            return response.json()
    except requests.ConnectionError as e:
        print('Error','-',tab,Type,':',e.args)

def save_data(tab,Type,data):
    data=pd.DataFrame(data=data,index=None)
    if Type=='':
        name=tab
    else:
        name=tab+'-'+Type
    location='/home/zhanglei/桌面/西甲/'+name+'.xlsx'
    print(location)
    data.to_excel(location)

tab_list=list(tabs.keys())
for tab in tab_list:
    for Type in tabs[tab]:
        try:
            if tab=='赛程':
                datas=get_data('赛程','')['data']
                data=[]
                for i in datas:
                    data+=i['list']
            else:
                data=get_data(tab,Type)['data']
            save_data(tab,Type,data)
            if Type=='':
                print(tab,'finished!')
            else:
                print(tab,'-',Type,'finished!')
        except:
            print(tab,'-',Type,'error')
            continue

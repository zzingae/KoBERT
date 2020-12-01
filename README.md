        emotion = self.df.loc[index, 'label']

        # 일상다반서 0, 이별(부정) 1, 사랑(긍정) 2로 레이블링
        if emotion==0:
            emotion_word = '일상 '
        elif emotion==1: # self.sp('이별') --> ['▁이', '별']
            emotion_word = '부정 '
        else:
            emotion_word = '사랑 '
        qtoks = self.sp(emotion_word + question)

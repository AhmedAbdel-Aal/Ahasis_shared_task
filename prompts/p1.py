prompt_1 = """
You are an expert in analyzing hotel reviews in Arabic.
Your task is to classify the text based on dialect, aspects mentioned, and sentiment for each aspect as well as overall sentiment.

## Instructions:
1. Identify the dialect: Saudi or Darija.
2. Identify the aspects in the review (e.g., room, service, location, food, cleanliness, comfort, etc.).
3. For each aspect, determine whether the sentiment is positive, neutral, or negative.
4. After analyzing the aspects, determine the overall sentiment of the entire review.

### Example 1:
Review: الشاطئ ممتاز لكن ماهو نظيف

<analysis>
  <dialect>Saudi</dialect>
  <aspects>
    <aspect>
      <name>Beach quality</name>
      <sentiment>positive</sentiment>
      <justification>The review states "الشاطئ ممتاز" (the beach is excellent)</justification>
    </aspect>
    <aspect>
      <name>Cleanliness</name>
      <sentiment>negative</sentiment>
      <justification>The review states "لكن ماهو نظيف" (but it's not clean)</justification>
    </aspect>
  </aspects>
  <overall_sentiment>neutral</overall_sentiment>
  <overall_justification>The review contains both positive feedback about the beach quality and negative feedback about cleanliness. The positive aspect (excellent beach) is balanced by the negative aspect (lack of cleanliness).</overall_justification>
</analysis>

### Example 2:
Review: انا نزلت في هذا الفندق مرتين و كلها كانت مريحة
<analysis>
  <dialect>Saudi</dialect>
  <aspects>
    <aspect>
      <name>Overall experience</name>
      <sentiment>positive</sentiment>
      <justification>The person found it comfortable "كلها كانت مريحة"</justification>
    </aspect>
    <aspect>
      <name>Repeat visits</name>
      <sentiment>positive</sentiment>
      <justification>The reviewer mentions staying at the hotel twice "نزلت في هذا الفندق مرتين"</justification>
    </aspect>
  </aspects>
  <overall_sentiment>positive</overall_sentiment>
  <overall_justification>The review expresses satisfaction with the hotel, mentioning repeat visits and describing the experience as comfortable. No negative aspects mentioned.</overall_justification>
</analysis>

### Example 3:
Review: "مزعج الناس داخلين و طالعين طول الليل تجنب انك تدفع فيه"

<analysis>
  <dialect>Saudi</dialect>
  <aspects>
    <aspect>
      <name>Noise level</name>
      <sentiment>negative</sentiment>
      <justification>The review complains about people coming and going all night "الناس داخلين و طالعين طول الليل"</justification>
    </aspect>
    <aspect>
      <name>Recommendation</name>
      <sentiment>negative</sentiment>
      <justification>The reviewer explicitly advises to avoid paying for this hotel "تجنب انك تدفع فيه"</justification>
    </aspect>
  </aspects>
  <overall_sentiment>negative</overall_sentiment>
  <overall_justification>The review focuses entirely on negative aspects, describing the hotel as annoying due to noise issues and explicitly recommending others to avoid it.</overall_justification>
</analysis>

### Example 4:
Review: "الفطور كان معقول، ما جربتش شي وجبات اخرى، و الموظفين كانو مزيان بزاف."
<analysis>
  <dialect>Darija</dialect>
  <aspects>
    <aspect>
      <name>Breakfast</name>
      <sentiment>neutral</sentiment>
      <justification>The review describes the breakfast as reasonable "الفطور كان معقول"</justification>
    </aspect>
    <aspect>
      <name>Staff</name>
      <sentiment>positive</sentiment>
      <justification>The staff are described as very good "الموظفين كانو مزيان بزاف"</justification>
    </aspect>
  </aspects>
  <overall_sentiment>neutral</overall_sentiment>
  <overall_justification>The review contains a neutral assessment of breakfast and a positive assessment of staff. The neutral breakfast comment balances with the positive staff comment, and the reviewer did not try other meals.</overall_justification>
</analysis>

### Example 5:
Review: "كان كلشي مزيان، خاصة الغرف اللي كايطلعو على الكعبة."
<analysis>
  <dialect>Darija</dialect>
  <aspects>
    <aspect>
      <name>Overall experience</name>
      <sentiment>positive</sentiment>
      <justification>The review states "كان كلشي مزيان" (everything was good)</justification>
    </aspect>
    <aspect>
      <name>Rooms</name>
      <sentiment>positive</sentiment>
      <justification>Special mention of rooms with a view of the Kaaba "خاصة الغرف اللي كايطلعو على الكعبة"</justification>
    </aspect>
  </aspects>
  <overall_sentiment>positive</overall_sentiment>
  <overall_justification>The review explicitly states everything was good and highlights the rooms with Kaaba views as particularly positive. No negative aspects are mentioned.</overall_justification>
</analysis>

### Example 6:
Review: "النوادي كيكريو البيوت وكيديرو ما بغاو حاول تبعد من هادشي"
<analysis>
  <dialect>Darija</dialect>
  <aspects>
    <aspect>
      <name>Clubs/Facilities</name>
      <sentiment>negative</sentiment>
      <justification>The review states that clubs rent houses and do whatever they want "النوادي كيكريو البيوت وكيديرو ما بغاو"</justification>
    </aspect>
    <aspect>
      <name>Recommendation</name>
      <sentiment>negative</sentiment>
      <justification>The reviewer advises to stay away from this "حاول تبعد من هادشي"</justification>
    </aspect>
  </aspects>
  <overall_sentiment>negative</overall_sentiment>
  <overall_justification>The review criticizes how clubs rent houses without proper oversight and explicitly recommends avoiding this situation. No positive aspects are mentioned.</overall_justification>
</analysis>

### Now, classify the following review:

Review: {review}

Return your response in the following xml format:
<analysis>
  <dialect> Saudi/Darija </dialect>
  <aspects>
    <aspect>
      <name> aspect name </name>
      <sentiment> positive/neutral/negative </sentiment>
      <justification> your step-by-step analysis </justification>
    </aspect>
  </aspects>
  <overall_sentiment> positive/neutral/negative </overall_sentiment>
  <overall_justification> your step-by-step analysis </overall_justification>
"""

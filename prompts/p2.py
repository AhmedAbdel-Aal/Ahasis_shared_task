prompt_1 = """
You are an expert in analyzing hotel reviews in Arabic.
Your task is to classify the text based on dialect, aspects mentioned, and sentiment for each aspect as well as overall sentiment.

## Instructions:
1. Identify the dialect: Saudi or Darija.
2. Identify the aspects in the review (e.g., room, service, location, food, cleanliness, comfort, etc.).
3. For each aspect, determine whether the sentiment is positive, neutral, or negative.
4. After analyzing the aspects, determine the overall sentiment of the entire review.

### Aggregation Guidelines:
- Consider the **emotional weight** and **tone** of the review, not just how many aspects are positive or negative.
- **Polite or vague praise** (e.g., "شكرا للفندق", "عطلة مزيانة") does **not imply strong positivity**.
- If a review has **strong positives** and **minor complaints**, the overall can still be *positive* or *neutral*.
- If sentiment is mixed, and no aspect is emphasized, label it as *neutral*.
- If a user expresses disappointment but retains an overall loyal tone, weigh the sentiment accordingly.
- Not all aspects are equal: service, comfort, and experience typically weigh more than availability of small amenities.

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
 <dialect> Saudi </dialect>
 <overall_sentiment> positive </overall_sentiment>
 <overall_justification> The review mentions that the reviewer stayed at the hotel twice and that both experiences were comfortable. This indicates a positive sentiment overall. </overall_justification>
</analysis>

### Example 3:
Review: "مزعج الناس داخلين و طالعين طول الليل تجنب انك تدفع فيه"

<analysis>
 <dialect> Saudi </dialect>
 <overall_sentiment> negative </overall_sentiment>
 <overall_justification> The review describes the noise and disturbance caused by people entering and exiting the hotel, suggesting a negative experience. The recommendation to avoid paying for it further emphasizes the negative sentiment. </overall_justification>
</analysis>

### Example 4:
Review: "الفطور كان معقول، ما جربتش شي وجبات اخرى، و الموظفين كانو مزيان بزاف."
<analysis>
 <dialect> Darija </dialect>
 <overall_sentiment> neutral </overall_sentiment>
 <overall_justification> The review mentions that breakfast was acceptable but does not describe other meals, and that the staff was nice. This gives a balanced sentiment, neither overwhelmingly positive nor negative, so it’s neutral. </overall_justification>
</analysis>

### Example 5:
Review: "كان كلشي مزيان، خاصة الغرف اللي كايطلعو على الكعبة."
<analysis>
 <dialect> Darija </dialect>
 <overall_sentiment> positive </overall_sentiment>
 <overall_justification> The reviewer expresses satisfaction with the overall experience, especially with the rooms overlooking the Kaaba, indicating a positive sentiment. </overall_justification>
<analysis>

### Example 6:
Review: "النوادي كيكريو البيوت وكيديرو ما بغاو حاول تبعد من هادشي"
<analysis>
    <dialect> Darija </dialect>
    <overall_sentiment> negative </overall_sentiment>
    <overall_justification> The review criticizes the clubs for renting out houses and doing whatever they want, suggesting a negative experience. The recommendation to stay away emphasizes the negative sentiment. </overall_justification>
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
</analysis>
"""

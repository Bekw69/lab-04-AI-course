"""Create the one-page hand-in from measured summary JSON (Arial, Swiss Academic)."""

import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

# Optional PDF dependencies live outside the model's verified environment.
sys.path.insert(0,str(Path('.tools/report').resolve()))
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from pypdf import PdfReader

from lab04_analysis import fmt


def main():
    root=Path('results/intel')
    s=json.loads((root/'metrics_summary.json').read_text(encoding='utf-8'))
    e=json.loads((root/'environment.json').read_text(encoding='utf-8'))
    target=Path('output/pdf/lab04_submission.pdf')
    target.parent.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont('Academic','C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('Academic-Bold','C:/Windows/Fonts/arialbd.ttf'))
    pdfmetrics.registerFontFamily('Academic',normal='Academic',bold='Academic-Bold',italic='Academic',boldItalic='Academic-Bold')
    ink,navy,border=colors.HexColor('#0F172A'),colors.HexColor('#1E3A8A'),colors.HexColor('#DBE2EA')
    body=ParagraphStyle('body',fontName='Academic',fontSize=10.2,leading=12.5,textColor=ink,spaceAfter=5)
    small=ParagraphStyle('small',parent=body,fontSize=9,leading=11,spaceAfter=6)
    heading=ParagraphStyle('heading',parent=body,fontName='Academic-Bold',fontSize=11,leading=13,spaceBefore=5,spaceAfter=4,textColor=navy)
    title=ParagraphStyle('title',parent=heading,fontSize=17,leading=20,spaceBefore=0,spaceAfter=8)
    table_text=ParagraphStyle('table',parent=body,fontSize=9.2,leading=11.4,spaceAfter=0)
    white=ParagraphStyle('white',parent=table_text,fontName='Academic-Bold',textColor=colors.white)
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=38,rightMargin=38,topMargin=31,bottomMargin=30,
                          title='Lab 04 - Invoice to JSON',author='Lab 04 measured CPU run')
    story=[]
    para=lambda text,style=body:Paragraph(text,style)
    def section(text):story.append(para(text,heading))
    def table(rows,widths):
        cells=[[para(escape(str(value)),white if i==0 else table_text) for value in row] for i,row in enumerate(rows)]
        item=Table(cells,colWidths=widths,hAlign='LEFT')
        item.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('GRID',(0,0),(-1,-1),.45,border),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([item,Spacer(1,5)])
    width=A4[0]-76
    story.append(para('Lab 04 | Invoice to JSON',title))
    story.append(para(f"Narxoz University. Собственный прогон: 48 изображений + 16 дополнительных.<br/>"
        f"Qwen2-VL-2B-Instruct; OpenVINO {e['versions']['openvino']}; Intel Core i5-1235U (CPU). "
        'Веса INT4/INT8, вычисления FP32; greedy, 512 новых токенов. Это квантизованный экспорт, а не float16-эталон.',small))

    section('Прогнозы и измерение')
    weakest=min(('kk','ru','en'),key=lambda lang:s['accuracy']['clean'][lang]['norm']['accuracy'])
    ocr_clean=s['accuracy']['clean']['all']['ocr']['accuracy']
    vlm_clean=s['accuracy']['clean']['all']['norm']['accuracy']
    ocr_bad=s['accuracy']['degraded']['all']['ocr']['accuracy']
    vlm_bad=s['accuracy']['degraded']['all']['norm']['accuracy']
    story.append(para('Ожидания после чтения README, до собственного прогона (не слепые): '
        'Predict 1 - больше ошибок в казахских названиях, сильнее падение OCR; clean около 90% OCR и 85% VLM. '
        f'Измерение: слабее всего clean/{weakest}; падение OCR {(ocr_clean-ocr_bad)*100:.1f} п.п., VLM {(vlm_clean-vlm_bad)*100:.1f} п.п. '
        'Predict 2 - OCR выше на clean, VLM на degraded. '
        f"Измерение: clean выше {'OCR' if ocr_clean>vlm_clean else 'VLM'}, degraded выше {'VLM' if vlm_bad>ocr_bad else 'OCR'}."))
    table([['Метод; 95% Wilson','Clean','Degraded'],
           ['OCR upper bound',fmt(s['accuracy']['clean']['all']['ocr']),fmt(s['accuracy']['degraded']['all']['ocr'])],
           ['VLM normalised',fmt(s['accuracy']['clean']['all']['norm']),fmt(s['accuracy']['degraded']['all']['norm'])]],
          [width*.27,width*.365,width*.365])

    section('Казахские названия и причины ошибок')
    pairs=s['confusions']['all']['all_letters'][:3]
    text='; '.join(f"{r['true']} ({r['true_codepoint']}) → {r['predicted']} ({r['predicted_codepoint']}): {r['count']}" for r in pairs)
    story.append(para('Три частые замены букв в kk, обе версии вместе (истина → модель): '+escape(text)+'. '
        'Одних замен недостаточно, чтобы отличить чтение от языкового смещения. Контроль: реальные слова и случайные строки из тех же букв при одинаковых шрифте, размере и качестве.'))

    section('Арифметическая проверка')
    rows=[['Скан','Пойманы неверные итоги','Отмечены правильные итоги','Прошли с ошибкой / все прошедшие']]
    for cond in ('clean','degraded'):
        a=s['arithmetic'][cond]
        rows.append([cond,f"{a['caught']}/{a['wrong_totals']}",f"{a['right_flagged']}/{a['right_totals']}",f"{a['passed_with_errors']}/{a['passed']}"])
    table(rows,[width*.16,width*.26,width*.27,width*.31])
    story.append(para('Одобрять оплату этими проверками нельзя: согласованные неверные числа и ошибки в названиях могут пройти обе проверки.'))

    section('Дополнительные задания и выбор системы')
    parts=[]
    for cond,v in s['letter_experiment'].items():
        a,b=v['before']['names'],v['after']['names']
        parts.append(f"{cond}: {a['fields']-a['correct']}/{a['fields']} → {b['fields']-b['correct']}/{b['fields']}")
    story.append(para('Advanced 5, неверные названия до → после: '+'; '.join(parts)+'. '
        'Изменение промпта само по себе не устанавливает причину ошибки. Core 1-3: все таблицы сегментации, дат и арифметики приведены в ноутбуке и приложении.'))
    story.append(para('Выбор: контроль качества снимка и повторный скан при размытии; OCR с парсером для читаемых шаблонов, VLM для исключений; '
        'проверка схемы, дат, БИН и арифметики, затем подтверждение человеком перед оплатой.'))
    story.append(para('OCR здесь измеряет наличие значений, а не готовый JSON. Поля и переводы одного счёта зависимы; Wilson-интервалы описательные. '
        'Версии, ревизия модели, исходные ответы, результаты по языкам и полный разбор: results/intel/analysis.md и environment.json.',small))

    doc.build(story)
    reader=PdfReader(target)
    if len(reader.pages)!=1:
        raise RuntimeError(f'Expected one-page submission; produced {len(reader.pages)} pages.')
    print(target,'- 1 page, embedded Unicode fonts')


if __name__=='__main__':main()

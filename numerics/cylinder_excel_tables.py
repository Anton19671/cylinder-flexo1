"""Common self-describing Excel tables for Cylinder-Flexo1.

Used by every workbook exporter. One quantity per column, an SI unit in
each heading, an unambiguous symbol/code, and a glossary in EVERY workbook.
No numerical solver, physical field or exported scalar is recalculated here.
"""
from __future__ import annotations

import math
import re

import numpy as np
import xlsxwriter


# (Russian caption, mathematical symbol, SI unit, unambiguous interpretation)
COLUMN_INFO = {
    "time_s": ("Время", "t", "с", "Физическое время, отсчитанное по временной сетке решателя."),
    "load_factor": ("Множитель нагрузки", "lambda_F", "1", "Безразмерный коэффициент приложенной нагрузки."),
    "w_m": ("Радиальное перемещение", "w", "м", "Радиальное перемещение точки срединной поверхности цилиндра."),
    "dw_dt_m_s": ("Радиальная скорость", "dw/dt", "м/с", "Скорость радиального перемещения выбранной точки."),
    "d2w_dt2_m_s2": ("Радиальное ускорение", "d2w/dt2", "м/с²", "Ускорение радиального перемещения выбранной точки."),
    "u_m": ("Продольное перемещение", "u", "м", "Перемещение вдоль оси цилиндра x."),
    "v_m": ("Окружное перемещение", "v", "м", "Перемещение вдоль окружной координаты s."),
    "v_tangential_m": ("Окружное перемещение", "v", "м", "Окружное перемещение точки срединной поверхности."),
    "psi0_v": ("Потенциал, нулевой коэффициент", "phi_0", "В", "Нулевой коэффициент разложения электрического потенциала по толщине."),
    "inner_w_m": ("Радиальное перемещение, внутренняя грань", "w_in", "м", "Перемещение на внутренней поверхности оболочки."),
    "outer_w_m": ("Радиальное перемещение, внешняя грань", "w_out", "м", "Перемещение на наружной поверхности оболочки."),
    "frequency_Hz": ("Частота спектрального отсчёта", "f", "Гц", "Частота соответствующего отсчёта дискретного спектра Фурье."),
    "frequency_hz": ("Собственная частота", "f_i", "Гц", "Собственная частота i-й формы колебаний."),
    "omega_rad_s": ("Круговая частота", "omega", "рад/с", "Круговая частота: omega = 2*pi*f."),
    "omega_squared_rad2_s2": ("Квадрат собственной частоты", "omega²", "рад²/с²", "Собственное значение обобщённой задачи K*phi=omega²*M*phi."),
    "amplitude_m": ("Амплитуда Фурье", "A(f)", "м", "Односторонняя амплитуда преобразования Фурье радиального перемещения."),
    "PSD_m2_per_Hz": ("Спектральная плотность мощности", "PSD_w", "м²/Гц", "Спектральная плотность мощности сигнала радиального перемещения."),
    "strobe_time_s": ("Время сечения Пуанкаре", "t_k", "с", "Временная отметка выборки с периодом внешнего воздействия."),
    "evolution_s": ("Время расхождения", "tau", "с", "Время эволюции расстояний между соседними траекториями."),
    "mean_log_distance": ("Логарифм среднего расстояния", "ln d", "1", "Средний логарифм расстояния в нормированном реконструированном пространстве."),
    "linear_fit": ("Линейная аппроксимация", "a*t+b", "1", "Аппроксимация среднего логарифма расстояния по времени."),
    "probe": ("Контрольная точка", "P", "—", "Уникальное имя контрольной точки на поверхности цилиндра."),
    "node_x": ("Продольный индекс узла", "i_x", "1", "Нулевой индекс узла вдоль оси цилиндра, как в исходном массиве."),
    "node_s": ("Окружной индекс узла", "i_s", "1", "Нулевой индекс узла по окружности, как в исходном массиве."),
    "ix": ("Продольный индекс узла", "i_x", "1", "Нулевой индекс узла по оси цилиндра."),
    "js": ("Окружной индекс узла", "i_s", "1", "Нулевой индекс узла по окружности."),
    "x_m": ("Продольная координата", "x", "м", "Координата вдоль оси цилиндра."),
    "s_m": ("Дуговая координата", "s", "м", "Координата по длине окружной дуги срединной поверхности."),
    "theta_rad": ("Угловая координата", "theta", "рад", "Угол по окружности, theta=s/R."),
    "theta_x_rad": ("Поворот директора вдоль x", "theta_x", "рад", "Первая независимая угловая координата FSDT или Reddy."),
    "theta_s_rad": ("Поворот директора вдоль s", "theta_s", "рад", "Вторая независимая угловая координата FSDT или Reddy."),
    "mode": ("Номер собственной формы", "i", "1", "Порядковый номер формы, отсортированной по неубыванию собственной частоты."),
    "index": ("Индекс", "i", "1", "Порядковый номер элемента данной таблицы."),
    "lyapunov_1_per_s": ("Показатель Ляпунова по QR", "lambda_i", "1/с", "Показатель, вычисленный только из переданных истинных карт касательной динамики."),
    "lambda_est_1_per_s": ("Оценка максимального показателя Ляпунова", "lambda_est", "1/с", "Оценка Розенштейна по одному сигналу; не полный спектр динамической системы."),
    "rosenstein_lambda_est_1_s": ("Оценка Розенштейна", "lambda_est", "1/с", "Оценка максимального показателя Ляпунова из скалярного временного сигнала."),
    "fit_R2": ("Качество линейной аппроксимации", "R²", "1", "Коэффициент детерминации линейного участка логарифмического расхождения."),
    "neighbor_pairs": ("Число пар соседей", "N_pairs", "1", "Число пар близких траекторий, использованных в оценке."),
    "delay_samples": ("Задержка вложения", "tau_embed", "отсчётов", "Временная задержка между координатами реконструкции в количестве шагов."),
    "theiler_samples": ("Окно Тайлера", "W_Theiler", "отсчётов", "Минимальное временное разнесение точек при поиске соседей."),
    "peak_time_s": ("Время локального максимума", "t_peak", "с", "Физический момент локального максимума радиального перемещения."),
    "radial_peak_m": ("Локальный максимум перемещения", "w_peak", "м", "Локальный максимум радиального перемещения после переходного участка."),
    "n_peaks": ("Число максимумов", "N_peak", "1", "Число локальных максимумов после исключения переходного участка."),
    "n_strobes": ("Число отсчётов Пуанкаре", "N_P", "1", "Число точек стробоскопического сечения."),
    "w_mean_m": ("Среднее радиальное перемещение", "mean(w)", "м", "Среднее радиальное перемещение на принятом участке истории."),
    "w_std_m": ("Стандартное отклонение перемещения", "std(w)", "м", "Стандартное отклонение радиального перемещения."),
    "estimate_status": ("Статус оценки", "status", "—", "Код применимости/качества скалярной оценки."),
    "theory": ("Теория оболочки", "theory", "—", "KL, FSDT, Reddy или CUF TE1–TE4."),
    "method": ("Пространственный метод", "method", "—", "Применённый численный метод."),
    "rank": ("Ранг МВИ", "R", "1", "Число членов разделённого представления МВИ."),
    "complete": ("Полная история расчёта", "complete", "—", "Признак наличия всех принятых временных шагов."),
    "accepted_time_steps": ("Число принятых шагов", "N_t", "1", "Количество принятых физических шагов динамики."),
    "dt_s": ("Шаг по времени", "dt", "с", "Длительность одного шага динамики."),
    "transient_fraction": ("Доля исключённого переходного участка", "alpha", "1", "Доля начала ряда, отброшенная при обработке."),
    "forcing_hz": ("Частота вынуждающего воздействия", "f_ext", "Гц", "Заданная частота внешнего воздействия; отсутствует для негармонических случаев."),
    "dominant_frequency_Hz": ("Доминирующая частота Фурье", "f_dom", "Гц", "Частота максимальной разрешённой амплитуды спектра."),
    "fft": ("Статус спектрального анализа", "FFT", "—", "Код наличия и достоверности БПФ."),
    "poincare": ("Статус сечения Пуанкаре", "P", "—", "Код наличия стробоскопического сечения."),
    "lyapunov": ("Статус оценки Ляпунова", "LE", "—", "Код наличия скалярной оценки."),
    "property": ("Код свойства расчёта", "property", "—", "Имя атрибута CASE или итогового отчёта; см. значение справа."),
    "setting": ("Название настройки", "setting", "—", "Код параметра численного исследования или обработки."),
    "value": ("Значение параметра", "value", "по строке", "Значение, тип и физическая единица зависят от названия параметра слева."),
    "workbook": ("Книга Excel", "file", "—", "Имя другого файла Excel из той же серии вывода."),
    "parameter_value": ("Значение параметра исследования", "p", "по CASE", "Значение независимого параметра серии; единица определяется исследуемым параметром CASE."),
    "n_modes": ("Число собственных форм", "N_mode", "1", "Количество сохранённых собственных форм."),
    "density_kg_m3": ("Плотность материала", "rho", "кг/м³", "Массовая плотность механической модели."),
    "E0_Pa": ("Модуль Юнга", "E", "Па", "Модуль Юнга материала."),
    "nu0": ("Коэффициент Пуассона", "nu", "1", "Коэффициент Пуассона материала."),
    "L_m": ("Длина цилиндра", "L", "м", "Длина цилиндрической оболочки вдоль оси."),
    "R_m": ("Радиус цилиндра", "R", "м", "Радиус срединной поверхности цилиндра."),
    "h_m": ("Толщина оболочки", "h", "м", "Полная толщина цилиндрической оболочки."),
    "nx": ("Число узлов по оси", "n_x", "1", "Число узлов вдоль оси цилиндра."),
    "ns": ("Число узлов по окружности", "n_s", "1", "Число узлов по окружности цилиндра."),
    "n_gauss": ("Число точек Гаусса", "n_g", "1", "Число квадратурных точек через толщину."),
    "support_left": ("Закрепление при x=0", "BC_left", "—", "Вид механического закрепления левого торца."),
    "support_right": ("Закрепление при x=L", "BC_right", "—", "Вид механического закрепления правого торца."),
    "code": ("Код параметра", "key", "—", "Машиночитаемое поле результирующего файла."),
    "description": ("Пояснение", "info", "—", "Физический смысл параметра."),
    "units": ("Единица измерения", "SI", "—", "Единица в Международной системе единиц либо обозначение безразмерности."),
    "data_file": ("Имя файла данных", "file", "—", "Имя файла с исходными числовыми данными."),
}


def _describe(code):
    name = str(code)
    found = COLUMN_INFO.get(name)
    if found is not None:
        return found
    electric_match = re.fullmatch(r"phi([0-4])_v", name)
    if electric_match:
        k = electric_match.group(1)
        return (f"Коэффициент электрического потенциала {k}", f"phi_{k}", "В",
                f"Коэффициент phi_{k} аппроксимации потенциала по толщине xi=2z/h.")
    match = re.fullmatch(r"([uvw])([0-4])_m", name)
    if match:
        symbol, k = match.groups()
        axis = {"u":"продольного", "v":"окружного", "w":"радиального"}[symbol]
        return (f"Коэффициент {axis} перемещения TE{k}", f"{symbol}_{k}", "м",
                f"Коэффициент {symbol}_{k} разложения CUF по безразмерной толщине xi=2z/h.")
    return (f"Поле {name}", name, "уточнить в CASE",
            "Пользовательский или дополнительный параметр; точное определение "
            "и единица должны браться из CASE/манифеста, автоматически не предполагаются.")


def _header(code):
    caption, symbol, unit, _ = _describe(code)
    unit_part = "" if unit in ("", "—") else f", {unit}"
    return f"{caption} ({symbol}{unit_part})"


class LabeledWorkbook(xlsxwriter.Workbook):
    """A regular xlsxwriter Workbook that always receives a column glossary."""
    def __init__(self, filename, options=None):
        super().__init__(filename, options)
        self._cylinder_columns = []
        self._cylinder_glossary_written = False

    def close(self):
        if not self._cylinder_glossary_written:
            self._cylinder_glossary_written = True
            self._write_glossary()
        return super().close()

    def _write_glossary(self):
        sh = self.add_worksheet("ОБОЗНАЧЕНИЯ")
        sh.hide_gridlines(2)
        sh.freeze_panes(1, 0)
        sh.set_zoom(85)
        header = self.add_format({"bold": True, "bg_color": "#DADADA",
                                  "border": 1, "text_wrap": True, "valign": "vcenter"})
        normal = self.add_format({"text_wrap": True, "valign": "top"})
        link = self.add_format({"underline": True, "font_color": "#333333",
                                "text_wrap": True, "valign": "top"})
        heads = ("Лист", "Столбец", "Код поля", "Обозначение", "Единица СИ",
                 "Физический смысл / пояснение")
        for col, label in enumerate(heads):
            sh.write(0, col, label, header)
        sh.set_row(0, 32)
        for col, width in enumerate((24, 29, 29, 20, 17, 92)):
            sh.set_column(col, col, width)
        for row, (sheet, key) in enumerate(self._cylinder_columns, start=1):
            caption, symbol, unit, description = _describe(key)
            safe_sheet = sheet.replace("'", "''")
            sh.write_url(row, 0, f"internal:'{safe_sheet}'!A1", link, sheet)
            sh.write(row, 1, caption, normal)
            sh.write_string(row, 2, str(key), normal)
            sh.write_string(row, 3, symbol, normal)
            sh.write_string(row, 4, unit, normal)
            sh.write_string(row, 5, description, normal)
            sh.set_row(row, 34)
        if self._cylinder_columns:
            sh.autofilter(0, 0, len(self._cylinder_columns), len(heads)-1)


def write_labeled_table(wb, name, columns, rows, *, chart=None, chart_title=None):
    """Write data rows with named/unit-labelled columns and Excel filtering.

    Values are *imported* solver results. None/NaN/inf are left blank, never
    silently converted to invented numbers. Embedded charts point to the data.
    """
    headers = tuple(str(key) for key in columns)
    if not headers or len(set(headers)) != len(headers):
        raise ValueError("Excel columns must be nonempty and unique")
    sh = wb.add_worksheet(name[:31])
    sh.hide_gridlines(2)
    sh.set_zoom(90)
    heading = wb.add_format({"bold": True, "bg_color": "#DADADA", "border": 1,
                             "text_wrap": True, "valign": "vcenter"})
    scientific = wb.add_format({"num_format": "0.000000E+00"})
    numeric = wb.add_format({"num_format": "0.000000"})
    integer = wb.add_format({"num_format": "0"})
    sh.set_row(0, 47)
    for col, key in enumerate(headers):
        caption, symbol, unit, info = _describe(key)
        sh.write_string(0, col, _header(key), heading)
        sh.write_comment(0, col,
            f"Код: {key}\nОбозначение: {symbol}\nЕдиница: {unit}\n"
            f"Описание: {info}")
        fmt = (integer if unit in ("отсчётов",) or key in
               ("mode", "index", "rank", "n_peaks", "n_strobes", "neighbor_pairs",
                "delay_samples", "theiler_samples", "node_x", "node_s",
                "ix", "js", "n_modes", "nx", "ns", "n_gauss", "accepted_time_steps")
               else numeric if key in ("frequency_hz", "frequency_Hz", "omega_rad_s",
                                      "time_s", "x_m", "s_m", "theta_rad")
               else scientific)
        sh.set_column(col, col, 29 if len(caption) < 33 else 36, fmt)
        if hasattr(wb, "_cylinder_columns"):
            wb._cylinder_columns.append((sh.name, key))
    count = 0
    for count, row in enumerate(rows, start=1):
        if count >= 1048576:
            raise ValueError("Excel supports at most 1048575 data rows per sheet")
        values = list(row)
        if len(values) != len(headers):
            raise ValueError(f"{name}: row {count} has {len(values)} values, expected {len(headers)}")
        for col, val in enumerate(values):
            if val is None:
                continue
            if isinstance(val, (bool, np.bool_)):
                sh.write_boolean(count, col, bool(val))
            elif isinstance(val, (int, float, np.integer, np.floating)):
                if math.isfinite(float(val)):
                    sh.write_number(count, col, float(val))
            else:
                sh.write_string(count, col, str(val))
    sh.freeze_panes(1, 0)
    if count:
        sh.autofilter(0, 0, count, len(headers)-1)
    if chart is not None and count >= 2:
        kind, xx, yy = chart
        if not (0 <= xx < len(headers) and 0 <= yy < len(headers)):
            raise ValueError("Excel chart indexes are outside table columns")
        plot = wb.add_chart({"type": "scatter", "subtype": "straight"
                             if kind == "line" else "marker"})
        plot.add_series({
            "name": _header(headers[yy]),
            "categories": [sh.name, 1, xx, count, xx],
            "values": [sh.name, 1, yy, count, yy],
            "line": {"color": "#333333", "width": 1.2},
            "marker": {"type": "none" if kind == "line" else "circle",
                       "size": 3, "border": {"color": "#333333"},
                       "fill": {"color": "#777777"}},
        })
        plot.set_title({"name": chart_title or name})
        plot.set_x_axis({"name": _header(headers[xx])})
        plot.set_y_axis({"name": _header(headers[yy])})
        plot.set_legend({"none": True})
        sh.insert_chart(2, len(headers)+1, plot)
    return sh


__all__ = ("COLUMN_INFO", "LabeledWorkbook", "write_labeled_table")

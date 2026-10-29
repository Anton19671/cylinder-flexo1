"""Numerical, student-ready modal plots data, exported without running the solver.

All tabulated values come from saved normalized eigenvectors and the already
recovered linear modal fields. 3D Cartesian coordinates use the reference
cylindrical basis (axial, circumferential, outward radial). Amplification is
visual only; no modal value is a physical forced-response amplitude.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import xlsxwriter

from cylinder_excel_tables import LabeledWorkbook, write_labeled_table

XYZ_FILE = "03_МОДАЛЬНЫЕ_ФОРМЫ_КООРДИНАТЫ_3D.xlsx"
MAPS_FILE = "04_МОДАЛЬНЫЕ_ПОЛЯ_КАРТЫ_2D.xlsx"
SECTIONS_FILE = "05_МОДАЛЬНЫЕ_СЕЧЕНИЯ_И_ПРОФИЛИ.xlsx"
ENERGY_FILE = "06_МОДАЛЬНЫЕ_ЭНЕРГИЯ_ЭКСТРЕМУМЫ.xlsx"
NODES_FILE = "07_МОДАЛЬНЫЕ_УЗЛОВЫЕ_ЛИНИИ.xlsx"
COMPARE_FILE = "08_СРАВНЕНИЕ_МВИ_МКР_ШАБЛОН.xlsx"
HARMONICS_FILE = "09_МОДАЛЬНЫЕ_ОКРУЖНЫЕ_ГАРМОНИКИ.xlsx"
EXTRA_WORKBOOKS = (XYZ_FILE, MAPS_FILE, SECTIONS_FILE,
                   ENERGY_FILE, NODES_FILE, COMPARE_FILE, HARMONICS_FILE)
SURFACES = ("INNER", "MIDDLE", "OUTER")
COMPONENTS = ("U", "V", "W")


def _sheet(book, name, headers, values):
    write_labeled_table(book, name, headers, values)


def _points(x, theta, *, seam=False):
    """Grid node list. For 3D and plots include the periodic seam once."""
    ns = len(theta)
    for ix, xpos in enumerate(x):
        for js in range(ns + int(seam)):
            actual = js % ns
            angle = float(theta[actual]) if js < ns else 2.*math.pi
            yield ix, js, actual, float(xpos), angle


def _properties(mode, frequency, radius):
    q = np.asarray(mode["displacement3_m"], float)
    if q.ndim != 4 or q.shape[0] != 3 or q.shape[-1] != 3:
        raise ValueError("Recovered modal displacement requires three surfaces and U,V,W")
    amp = float(np.max(np.linalg.norm(q, axis=-1)))
    if not np.isfinite(amp):
        raise ValueError("Non-finite modal displacement")
    # Only a separately labeled display shape is visually amplified.
    return min(1.e6, .12 * radius / amp) if amp > 0 else 0.


def _xyz_rows(mode_number, x, theta, radius, physics, frequency):
    d = physics["displacement3_m"]
    zvals = physics["z_m"]
    scale = _properties(physics, frequency, radius)
    for iz, name in enumerate(SURFACES):
        z = float(zvals[iz])
        for ix, js, j0, axial, angle in _points(x, theta, seam=True):
            u, v, w = map(float, d[iz, ix, j0])
            rad = radius + z
            c, sn = math.cos(angle), math.sin(angle)
            X0,Y0,Z0 = axial,rad*c,rad*sn
            # Unit scale is the stored mass-normalized shape, not an actual
            # motion. The display scale is explicitly marked separately.
            def moved(factor):
                return (axial+factor*u,
                        (rad+factor*w)*c-factor*v*sn,
                        (rad+factor*w)*sn+factor*v*c)
            X1,Y1,Z1 = moved(1.)
            Xp,Yp,Zp = moved(scale)
            yield (mode_number,name,ix,js,int(js==len(theta)),
                   axial,radius*angle,angle,z,u,v,w,
                   X0,Y0,Z0,X1,Y1,Z1,Xp,Yp,Zp,scale)


def export_xyz(root, x, theta, radius, frequency, physical):
    with LabeledWorkbook(str(root/XYZ_FILE)) as book:
        _sheet(book,"КООРДИНАТЫ_XYZ",(
            "mode","surface","ix","js","periodic_seam",
            "x_m","s_m","theta_rad","z_m",
            "u_shape","v_shape","w_shape",
            "X0_m","Y0_m","Z0_m",
            "X1_m","Y1_m","Z1_m",
            "Xplot_m","Yplot_m","Zplot_m","display_scale"),
            (row for i,phys in enumerate(physical,1)
             for row in _xyz_rows(i,x,theta,radius,phys,frequency[i-1])))
        _sheet(book,"ПОДСКАЗКА_3D",("setting","value"),(
            ("coordinate_system","X along axis; Y,Z Cartesian cross-section"),
            ("surface","INNER z=-h/2; MIDDLE z=0; OUTER z=h/2"),
            ("periodic_seam","One duplicated theta=2*pi node per axial station, used to close 3D mesh"),
            ("X0,Y0,Z0","Undeformed cylinder coordinates at each z"),
            ("X1,Y1,Z1","Coordinates with the saved mass-normalized modal vector at factor 1"),
            ("Xplot,Yplot,Zplot","For visual plotting only; amplified by display_scale"),
            ("display_scale","Display magnification is not the physical vibration amplitude"),
            ("grid","Use mode+surface, connect ix/js in order; do not connect different modes"),
        ))


def export_maps(root, x, theta, physical):
    # This workbook is intended for direct Excel surface/contour plots: each
    # sheet is a rectangular matrix, not a mixed field/coordinate database.
    with xlsxwriter.Workbook(str(root/MAPS_FILE)) as book:
        head = book.add_format({"bold":True,"bg_color":"#E1E5E8",
                                "text_wrap":True,"valign":"vcenter"})
        fmt = book.add_format({"num_format":"0.000000E+00"})
        index = book.add_worksheet("ИНДЕКС")
        index.hide_gridlines(2)
        index.write_row(0,0,("Лист","Форма","Слой","Поле","X → столбцы","θ → строки"),head)
        index.set_column(0,0,21)
        index.set_column(1,5,24)
        line = 1
        for i,mode in enumerate(physical,1):
            d = np.asarray(mode["displacement3_m"],float)
            for iz,surface in enumerate(SURFACES):
                for axis,name in enumerate(COMPONENTS):
                    tab = f"M{i:02d}_{surface[:3]}_{name}"
                    sh = book.add_worksheet(tab)
                    sh.hide_gridlines(2)
                    sh.freeze_panes(1,1)
                    sh.set_column(0,0,17)
                    sh.set_column(1,len(x),16,fmt)
                    sh.write(0,0,"θ [rad] / x [m]",head)
                    for ix,position in enumerate(x):
                        sh.write_number(0,ix+1,float(position),head)
                    for js in range(len(theta)+1):
                        jj=js % len(theta)
                        angle = float(theta[jj]) if js<len(theta) else 2*math.pi
                        sh.write_number(js+1,0,angle,fmt)
                        sh.write_row(js+1,1,[float(d[iz,ix,jj,axis])
                                            for ix in range(len(x))],fmt)
                    index.write_row(line,0,(tab,i,surface,name,"x_m","theta_rad"))
                    line+=1
        index.autofilter(0,0,line-1,5)
        notice=book.add_worksheet("ПОЯСНЕНИЯ")
        notice.set_column(0,0,25);notice.set_column(1,1,110)
        notes=(
            ("Что изображено","Массо-нормированные собственные формы U,V,W; это не амплитуды вынужденного движения"),
            ("Строки","Окружной угол theta [рад], замкнутый периодический шов theta=2*pi"),
            ("Столбцы","Продольная координата x [м]"),
            ("M01_INN_U","Форма 1, внутренняя поверхность, продольное перемещение U"),
            ("M01_MID_V","Форма 1, срединная поверхность, окружное перемещение V"),
            ("M01_OUT_W","Форма 1, наружная поверхность, радиальное перемещение W"),
            ("График","Стройте Surface/Contour по прямоугольному диапазону из выбранного листа"),
            ("Шаг сетки","Координаты сетки исходного расчёта; интерполяция не добавляется"),
        )
        for j,row in enumerate(notes):notice.write_row(j,0,row)


def _nearest_angles(theta):
    ids = [int(np.argmin(np.abs((theta-a+np.pi)%(2*np.pi)-np.pi)))
           for a in (0.,np.pi/2,np.pi,3*np.pi/2)]
    return list(dict.fromkeys(ids))


def _nearest_axial(x):
    target = (float(x[0]),float(x[0]+.25*(x[-1]-x[0])),
              float(x[0]+.5*(x[-1]-x[0])),
              float(x[0]+.75*(x[-1]-x[0])),float(x[-1]))
    return list(dict.fromkeys(int(np.argmin(np.abs(x-a))) for a in target))


def _dense_thickness(fields, theory, thickness, ix, js, q3):
    """Exact saved CUF polynomial; otherwise only physically recovered z."""
    name=str(theory).upper()
    if name.startswith("CUF_TE") and name[-1] in "1234":
        order=int(name[-1])+1
        samples=np.linspace(-thickness/2.,thickness/2.,21)
        coeffs=np.vstack((fields[0:order,ix,js],
                          fields[order:2*order,ix,js],
                          fields[2*order:3*order,ix,js]))
        basis=np.stack([(2*z/thickness)**np.arange(order) for z in samples])
        vals=basis @ coeffs.T
        return samples, vals, "EXACT_CUF_POLYNOMIAL"
    return np.array([-thickness/2,0.,thickness/2]),q3, "RECOVERED_3_STATIONS"


def export_sections(root, x, theta, radius, thickness, fields, theory, physical):
    longitudinal=[]; circumference=[]; through=[]
    axial_nodes=_nearest_axial(x)
    angular_nodes=_nearest_angles(theta)
    for i,(mode,raw) in enumerate(zip(physical,fields),1):
        d=mode["displacement3_m"]
        for iz,surface in enumerate(SURFACES):
            z=float(mode["z_m"][iz])
            for js in angular_nodes:
                for ix,axial in enumerate(x):
                    longitudinal.append((i,surface,ix,js,float(axial),
                                         radius*float(theta[js]),float(theta[js]),z,
                                         *map(float,d[iz,ix,js])))
            for ix in axial_nodes:
                for js in range(len(theta)+1):
                    jj=js%len(theta)
                    angle=float(theta[jj]) if js<len(theta) else 2.*math.pi
                    circumference.append((i,surface,ix,js,float(x[ix]),radius*angle,
                                          angle,z,*map(float,d[iz,ix,jj])))
        for ix in axial_nodes:
            for js in angular_nodes:
                zlist,vals,method=_dense_thickness(
                    raw,theory,thickness,ix,js,d[:,ix,js,:])
                for z,v in zip(zlist,vals):
                    through.append((i,ix,js,float(x[ix]),
                                    float(theta[js]),float(z),method,*map(float,v)))
    header=("mode","surface","ix","js","x_m","s_m","theta_rad","z_m",
            "u_shape","v_shape","w_shape")
    with LabeledWorkbook(str(root/SECTIONS_FILE)) as book:
        _sheet(book,"ПРОДОЛЬНЫЕ_X",header,longitudinal)
        _sheet(book,"ОКРУЖНЫЕ_THETA",header,circumference)
        _sheet(book,"ПРОФИЛИ_ПО_Z",(
            "mode","ix","js","x_m","theta_rad","z_m",
            "reconstruction_method","u_shape","v_shape","w_shape"),through)
        _sheet(book,"ПОДСКАЗКА_СЕЧЕНИЯ",("setting","value"),(
            ("longitudinal","For each mode, surface and theta, plot U/V/W against x"),
            ("circumferential","For each mode, surface and x, plot U/V/W against theta"),
            ("through thickness","For each mode/x/theta, plot U/V/W against z"),
            ("CUF_TE1..4","21 thickness samples from the saved CUF polynomial"),
            ("non-CUF","Only original physically recovered thickness stations; no invented interpolation"),
            ("normalization","Eigenvector magnitudes are arbitrary/mass-normalized"),
        ))


def _von_mises(stress):
    sx,sy,sz,txy,txz,tyz = map(float,stress)
    return float(np.sqrt(max(0.,.5*((sx-sy)**2+(sy-sz)**2+(sz-sx)**2)
                                 +3.*(txy*txy+txz*txz+tyz*tyz))))


def export_energy(root,x,theta,physical):
    energy=[]; extrema=[]
    for i,item in enumerate(physical,1):
        d=np.asarray(item["displacement3_m"])
        strain=np.asarray(item["strain6"])
        stress=np.asarray(item["stress6_pa"])
        energy_density=np.asarray(item["quadratic_enthalpy_j_per_m3"])
        for iz,surface in enumerate(SURFACES):
            z=float(item["z_m"][iz])
            for ix,axial in enumerate(x):
                for js,angle in enumerate(theta):
                    energy.append((i,surface,ix,js,float(axial),float(angle),z,
                                   float(energy_density[iz,ix,js]),
                                   _von_mises(stress[iz,ix,js])))
            for axis,component in enumerate(COMPONENTS):
                field=d[iz,:,:,axis]
                ix,js=np.unravel_index(np.argmax(np.abs(field)),field.shape)
                extrema.append((i,surface,component,"displacement_shape",
                                float(field[ix,js]),float(x[ix]),float(theta[js]),z))
            for label,kind,values in (
                    ("E_XX","strain",strain[iz,:,:,0]),
                    ("E_SS","strain",strain[iz,:,:,1]),
                    ("E_XS","strain",strain[iz,:,:,3]),
                    ("S_XX","stress_pa",stress[iz,:,:,0]),
                    ("S_SS","stress_pa",stress[iz,:,:,1]),
                    ("S_XS","stress_pa",stress[iz,:,:,3]),
                    ("ENERGY","enthalpy_j_m3",energy_density[iz])):
                ix,js=np.unravel_index(np.argmax(np.abs(values)),values.shape)
                extrema.append((i,surface,label,kind,float(values[ix,js]),
                                float(x[ix]),float(theta[js]),z))
    with LabeledWorkbook(str(root/ENERGY_FILE)) as book:
        _sheet(book,"ЭНЕРГИЯ_И_VM",(
            "mode","surface","ix","js","x_m","theta_rad","z_m",
            "quadratic_enthalpy_j_m3","von_mises_shape_pa"),energy)
        _sheet(book,"ЭКСТРЕМУМЫ",(
            "mode","surface","component","quantity","signed_extremum",
            "x_m","theta_rad","z_m"),extrema)
        _sheet(book,"ПРИМЕЧАНИЯ",("setting","value"),(
            ("energy","Quadratic electromechanical enthalpy from saved linearized modal state"),
            ("stress","Linearized stress shape; modal amplitudes are not physical stress response"),
            ("VM","Equivalent stress derived from saved six stress-shape components"),
            ("extremum","Maximum absolute value, storing actual sign and its node coordinates"),
            ("not physical peak","No external load/forced-response amplitude was specified"),
        ))


def _zeros(a,b,tol):
    if abs(a)<=tol and abs(b)<=tol:
        return None
    if abs(a)<=tol:return 0.
    if abs(b)<=tol:return 1.
    if (a>0) == (b>0):return None
    return float(abs(a)/(abs(a)+abs(b)))


def export_nodes(root,x,theta,physical):
    nodes=[]
    for mode_index,mode in enumerate(physical,1):
        d=mode["displacement3_m"]
        for iz,surface in enumerate(SURFACES):
            for k,name in enumerate(COMPONENTS):
                a=d[iz,:,:,k]
                maxv=float(np.max(np.abs(a)))
                if maxv==0:continue
                tol=maxv*1.e-9
                for ix in range(len(x)):
                    for js in range(len(theta)):
                        if ix+1<len(x):
                            t=_zeros(float(a[ix,js]),float(a[ix+1,js]),tol)
                            if t is not None:
                                nodes.append((mode_index,surface,name,"X_EDGE",
                                              float(x[ix]+t*(x[ix+1]-x[ix])),
                                              float(theta[js]),int(ix),int(js),
                                              float(t)))
                        jn=(js+1)%len(theta)
                        right=(float(theta[jn]) if jn>js else 2.*math.pi)
                        t=_zeros(float(a[ix,js]),float(a[ix,jn]),tol)
                        if t is not None:
                            nodes.append((mode_index,surface,name,"THETA_EDGE",
                                          float(x[ix]),
                                          float(theta[js]+t*(right-theta[js])),
                                          int(ix),int(js),float(t)))
    with LabeledWorkbook(str(root/NODES_FILE)) as book:
        _sheet(book,"ПЕРЕСЕЧЕНИЯ_НУЛЯ",(
            "mode","surface","component","edge_direction","x_m",
            "theta_rad","ix","js","edge_fraction"),nodes)
        _sheet(book,"ПОЯСНЕНИЯ",("setting","value"),(
            ("definition","Linear interpolation of zero crossing between adjacent saved grid nodes"),
            ("zero line","Points on grid edges; reconstruct contours using U/V/W map matrices"),
            ("periodicity","Theta edge between last node and first node closes at 2*pi"),
            ("boundaries","All-zero edges are skipped; do not mistake clamped boundary for internal nodal lines"),
            ("resolution","Nodal lines narrower than the spatial grid cannot be recovered"),
        ))


def export_comparison_template(root,frequency,method,summary):
    method=str(method).upper()
    with xlsxwriter.Workbook(str(root/COMPARE_FILE)) as book:
        header=book.add_format({"bold":True,"bg_color":"#DDDDDD","text_wrap":True})
        inputfmt=book.add_format({"font_color":"#0000FF","num_format":"0.00000000"})
        number=book.add_format({"num_format":"0.00000000"})
        percentage=book.add_format({"num_format":"0.0000"})
        sheet=book.add_worksheet("ЧАСТОТЫ_МВИ_МКР")
        sheet.freeze_panes(1,1)
        sheet.hide_gridlines(2)
        sheet.set_column(0,0,12)
        sheet.set_column(1,3,25)
        sheet.set_column(4,4,23)
        sheet.set_column(5,6,22)
        sheet.write_row(0,0,("Форма","f_МВИ, Гц","f_МКР, Гц",
                             "(МВИ-МКР)/МКР, %","|Δ|, %",
                             "Источник МВИ","Источник МКР"),header)
        for idx,f in enumerate(frequency,1):
            j=idx
            sheet.write_number(j,0,idx)
            if method=="MVI":
                sheet.write_number(j,1,float(f),number)
                sheet.write_blank(j,2,None,inputfmt)
                sheet.write_string(j,5,"Текущий расчёт МВИ")
            else:
                sheet.write_blank(j,1,None,inputfmt)
                sheet.write_number(j,2,float(f),number)
                sheet.write_string(j,6,"Текущий расчёт МКР")
            line=j+1
            sheet.write_formula(j,3,
                f'=IF(OR(NOT(ISNUMBER(B{line})),NOT(ISNUMBER(C{line})),C{line}=0),"",100*(B{line}-C{line})/C{line})',
                percentage)
            sheet.write_formula(j,4,f'=IF(ISNUMBER(D{line}),ABS(D{line}),"")',percentage)
        sheet.autofilter(0,0,len(frequency),6)
        note=book.add_worksheet("ИНСТРУКЦИЯ")
        note.set_column(0,0,28);note.set_column(1,1,105)
        info=(
            ("Текущий метод",method),
            ("Статус","Частоты отсутствующего метода намеренно оставлены пустыми"),
            ("Для сравнения","Заполнить СИНИЕ пустые ячейки частотами другого метода"),
            ("Сопоставление","Одинаковая геометрия, материал, сетка, краевые условия и число мод"),
            ("Кратные формы","Сопоставлять с учётом независимости собственных форм, не только порядка частот"),
            ("Разность","100*(f_МВИ - f_МКР)/f_МКР"),
            ("Не подставлять","Не принимать частоты разных сеток/случаев за эталонные"),
        )
        for i,row in enumerate(info):note.write_row(i,0,row)


def export_harmonics(root,x,theta,physical):
    rows=[]
    n=len(theta)
    for i,mode in enumerate(physical,1):
        d=mode["displacement3_m"]
        for iz,surface in enumerate(SURFACES):
            for axis,component in enumerate(COMPONENTS):
                fft=np.fft.rfft(d[iz,:,:,axis],axis=1)
                for ix,axial in enumerate(x):
                    for k,val in enumerate(fft[ix]):
                        mult=1. if k==0 or (n%2==0 and k==n//2) else 2.
                        rows.append((i,surface,component,ix,float(axial),k,
                                     float(mult*np.abs(val)/n),
                                     float(np.degrees(np.angle(val))),
                                     float(np.real(val)/n),
                                     float(np.imag(val)/n)))
    with LabeledWorkbook(str(root/HARMONICS_FILE)) as book:
        _sheet(book,"СПЕКТР_ПО_ОКРУЖНОСТИ",(
            "mode","surface","component","ix","x_m","wave_number",
            "harmonic_amplitude","phase_deg",
            "fourier_real","fourier_imag"),rows)
        _sheet(book,"ОПИСАНИЕ",("setting","value"),(
            ("transform","Discrete real FFT over original periodic theta nodes"),
            ("amplitude","Single-sided real Fourier amplitude per x station"),
            ("phase","Phase of FFT coefficient, degrees"),
            ("normalization","Saved eigenvector scale, not actual motion amplitude"),
            ("wave_number","Circumferential harmonic k; number of angular waves"),
            ("interpretation","Pairing sin/cos phase-shifted shapes requires independent vector check"),
        ))


def export_modal_student_data(root, summary, x, s, frequency, fields, physical):
    """Seven independent numeric books, ready for Excel/Python/Origin/MATLAB."""
    root=Path(root)
    frequency=np.asarray(frequency,float)
    fields=np.asarray(fields,float)
    x=np.asarray(x,float)
    s=np.asarray(s,float)
    radius=float(summary["R_m"]) if "R_m" in summary else None
    # Geometry is taken from the saved input CASE, never guessed from s max.
    import json
    case=json.loads((root/"REFERENCE_CASE.json").read_text(encoding="utf-8"))
    radius=float(case["geometry"]["R"])
    thickness=float(case["geometry"]["h"])
    theta=s/radius
    if not (radius>0 and thickness>0 and len(x)>=2 and len(theta)>=3 and
            len(frequency)==len(physical)==len(fields) and
            np.all(np.diff(x)>0) and np.all(np.diff(theta)>0)):
        raise ValueError("Modal student export input shape/geometry mismatch")
    export_xyz(root,x,theta,radius,frequency,physical)
    export_maps(root,x,theta,physical)
    export_sections(root,x,theta,radius,thickness,fields,
                    str(summary["theory"]),physical)
    export_energy(root,x,theta,physical)
    export_nodes(root,x,theta,physical)
    export_comparison_template(root,frequency,summary.get("method",""),
                               summary)
    export_harmonics(root,x,theta,physical)
    guide = root / "00_ЧИТАТЬ_ПЕРЕД_ПОСТРОЕНИЕМ_ГРАФИКОВ.txt"
    guide.write_text(
        "MODAL DATA / ДАННЫЕ ДЛЯ ГРАФИКОВ\n"
        "Все x, s, z и частоты в СИ. Theta в радианах.\n"
        "01: модальные характеристики, MAC, массы; 02: поля, деформации, напряжения.\n"
        "03: XYZ 3D, включая исходную и условно усиленную форму.\n"
        "04: готовые матрицы U/V/W по трем поверхностям для contour/surface.\n"
        "05: продольные, окружные сечения и профили по толщине.\n"
        "06: электромеханическая энтальпия и экстремумы; 07: точки узловых линий.\n"
        "08: шаблон МВИ/МКР, пустые частоты другого метода НЕ являются результатами.\n"
        "09: амплитуды и фазы окружных гармоник.\n"
        "Собственные векторы массо-нормированы: это НЕ физические амплитуды.\n"
        "Декартовы координаты Xplot/Yplot/Zplot усилены только для наглядности.\n"
        "Строить графики по одной форме и одной поверхности; не смешивать.\n",
        encoding="utf-8")
    return list(EXTRA_WORKBOOKS)


__all__=("export_modal_student_data","EXTRA_WORKBOOKS")

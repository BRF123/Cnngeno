#coding:utf-8

import pysam
from PIL import Image

def rearrange_string(read):
	bases=read.query_sequence
	new_bases=''
	new_base_quality=[]
	
	bases_checked=0
	read_len_count=0
	for cigar_portion in read.cigartuples:
		if cigar_portion[0]==0:
			cigar_base = bases[bases_checked:(cigar_portion[1]+bases_checked)]
			new_bases=new_bases+cigar_base
			for M_num in range(cigar_portion[1]):
				new_base_quality.append( min(read.query_alignment_qualities[read_len_count],read.query_qualities[read_len_count]) )
				read_len_count=read_len_count+1
			bases_checked=bases_checked+cigar_portion[1]

		elif cigar_portion[0]==1:
			bases_checked=bases_checked+cigar_portion[1]
			for I_num in range(cigar_portion[1]):
				read_len_count=read_len_count+1
		elif cigar_portion[0]==2:
			cigar_base=''
			for  i in range(cigar_portion[1]):
				cigar_base=cigar_base+'d'
				new_base_quality.append(0)
			new_bases=new_bases+cigar_base
		elif cigar_portion[0]==4 :
			cigar_base=''
			for  i in range(cigar_portion[1]):
				cigar_base=cigar_base+'s'
				new_base_quality.append(-1)
			new_bases=new_bases+cigar_base
			bases_checked=bases_checked+cigar_portion[1]
		elif cigar_portion[0]==5 :
			cigar_base=''
			for  i in range(cigar_portion[1]):
				cigar_base=cigar_base+'s'
				new_base_quality.append(-1)
			new_bases=new_bases+cigar_base
	return new_bases,new_base_quality

def read_can_shown(read,scan_l_pos,scan_r_pos):
	read_pos1=read.reference_start
	read_pos2=read.reference_start+read_infered_len(read)
	if (read_pos2 > scan_l_pos) and (read_pos1 < scan_r_pos):
		res=True
		for cigar_portion in read.cigartuples:
			if not ((cigar_portion[0]==0) or (cigar_portion[0]==1) or (cigar_portion[0]==2) or (cigar_portion[0]==4) or (cigar_portion[0]==5)):
				res = False
		return res
	else:
		return False

def read_corner_shown(read,scan_l_pos,scan_r_pos,new_bases):
	read_pos1=read.reference_start
	read_pos2=read.reference_start+read_infered_len(read)
	if (read_pos1 < scan_l_pos) and (read_pos2 > scan_l_pos):
		if ('A' or 'G' or 'C' or 'T' or 'a' or 'g' or 'c' or 't') in new_bases[(scan_l_pos-read_pos1):len(new_bases)] :
			return True
		else:
			return False
	if (read_pos1 < scan_r_pos) and (read_pos2 > scan_r_pos):
		if ('A' or 'G' or 'C' or 'T' or 'a' or 'g' or 'c' or 't') in new_bases[0:(scan_r_pos-read_pos1)] :
			return True
		else:
			return False
	if (read_pos1 >= scan_l_pos) and (read_pos2 <= scan_r_pos):
		return True

def read_infered_len(read):
	infer_len=0
	for cigar_portion in read.cigartuples:
		if ((cigar_portion[0]==0) or (cigar_portion[0]==2) or (cigar_portion[0]==4)):
			infer_len=infer_len+cigar_portion[1]
	return infer_len

def is_empty(read_list):
	tag=True
	for li in read_list:
		if li!=[]:
			tag=False
			break
	return tag

def get_shortest_tail_row(read_list,scan_r_pos):
	if is_empty(read_list):
		return 0
	else:
		tail=scan_r_pos
		short_row=0
		for i in range(len(read_list)):
			if read_list[i]!=[]:
				if tail > read_list[i][-1][1] :
					tail= read_list[i][-1][1]
					short_row=i
		return short_row

def find_next_empty_row(read_list):
	row=0
	for i in range(len(read_list)):
		if read_list[i]==[]:
			row=i
			break
	return row

def read_to_dictionary(read_package, scan_r_pos,height):
	dictionary={}
	read_list=[0 for i in range(height)]
	for i in range(height):
		read_list[i]=[]
	row_ptr=0
	for base_and_read in read_package:
		if row_ptr<height:
			base=base_and_read[0]
			quality=base_and_read[1]
			read=base_and_read[2]
			
			if read.cigartuples[0][0]==4 :
				read_pos1=read.reference_start - read.cigartuples[0][1]
				read_pos2=read_pos1+read_infered_len(read)
			else:
				read_pos1=read.reference_start
				read_pos2=read_pos1+read_infered_len(read)
			if read.is_paired: 
				is_concordant=read.is_proper_pair
				if 'S' in read.cigarstring:
					is_clipped=True
				else:
					is_clipped=False

				if is_empty(read_list):
					read_list[row_ptr].append( (read_pos1,read_pos2,base,quality,is_clipped,is_concordant) )
				else:
					row_ptr = get_shortest_tail_row(read_list,scan_r_pos)
					if read_pos1 >= read_list[row_ptr][-1][1]:
						read_list[row_ptr].append( (read_pos1,read_pos2,base,quality,is_clipped,is_concordant) )
					else:
						row_ptr=find_next_empty_row(read_list)
						read_list[row_ptr].append( (read_pos1,read_pos2,base,quality,is_clipped,is_concordant))

	for i in range(height):
		dictionary[i]=read_list[i]
	return dictionary

def draw_pgn(which_bp,dic,width,height,scan_l_pos,scan_r_pos,img_name):
	newIm = Image.new ("RGB", (width,height),(255,255,255))
	for key in range(height):
		for read_tuple in dic[key]:
			read_pos1=read_tuple[0]
			read_pos2=read_tuple[1]
			base=read_tuple[2]
			quality=read_tuple[3]
			is_clipped=read_tuple[4]
			is_concordant=read_tuple[5]
			col=read_pos1-scan_l_pos
			index_in_read=0
			for i in range(len(base)):
				if col >=0  and col < width :
					row=key
					red,green,blue=get_RGB(which_bp,base[index_in_read],quality[index_in_read],is_clipped,is_concordant)
					newIm.putpixel((col,row),(red,green,blue))
					index_in_read=index_in_read+1
					col=col+1
				elif col<0:
					index_in_read=index_in_read+1
					col=col+1
	newIm.save(img_name,"PNG")

def get_RGB(which_bp,base,quality,is_clipped,is_concordant):
	if is_clipped :
		red=255
		green=0
		blue=0
		if quality>1:
			green=green+255-6*quality
			blue=blue+255-6*quality
			return red, green, blue
		elif quality== 0 or quality== -1 :
			return 255,255,255
	elif is_concordant:
		red=0
		green=255
		blue=0
		if quality>1:
			red=red+255-6*quality
			blue=blue+255-6*quality
			return red, green, blue
		elif quality==0 or quality== -1:
			return 255,255,255
	elif not is_concordant:
		red=0
		green=0
		blue=255
		if quality>1:
			red=red+255-6*quality
			green=green+255-6*quality
			return red, green, blue
		elif quality==0 or quality== -1:
			return 255,255,255
	
def get_range(b1,b2):
	tmp=str(abs(b2-b1))
	high_bit=int(tmp[0])+1
	new_tmp=str(high_bit)
	for i in range(1,len(tmp)):
		new_tmp=new_tmp+'0'
	new_tmp=int(new_tmp)
	width= new_tmp+200
	height=50
	scan_l_pos= int(b1-(new_tmp-(b2-b1))/2-200)
	scan_r_pos=scan_l_pos+width

	return width,height,scan_l_pos,scan_r_pos

def main():

	vcf_path=' '
	bam_path=' '
	
	for line in open(vcf_path):
		print (line)
		line=line.strip('\n')
		tmp=line.split(' ')
		
	#	chrom=str(tmp[0])
		chrom=' '
		bk1=int(tmp[0])
		bk2=int(tmp[1])

		label='1'

		width=int(bk2-bk1+200)
		height=100
		scan_l_pos=bk1-100
		scan_r_pos=bk2+100

		samfile = pysam.AlignmentFile(bam_path,"rb")
		
		img_tmp=' '
		img_name=img_tmp+tmp[0]+'_'+tmp[1]+'.'+label+'.cover.png'
		#scan_l_pos=bk1-100
		#scan_r_pos=bk2+100
		
		read_package=[]
		c=0

		for read in samfile.fetch(chrom, scan_l_pos,scan_r_pos):
			if (read.cigarstring != None) and read_can_shown(read,scan_l_pos,scan_r_pos) and read.mapping_quality >10:
				new_bases,new_base_quality=rearrange_string(read)
				if read_corner_shown(read,scan_l_pos,scan_r_pos,new_bases):
					read_package.append((new_bases,new_base_quality,read))
					c=c+1
		dic=read_to_dictionary(read_package, scan_r_pos,height)
		print ('cover: ok before drawn')
		if not c==0 :
			draw_pgn('cover',dic,width,height,scan_l_pos,scan_r_pos,img_name)

if __name__ == '__main__':
    main()

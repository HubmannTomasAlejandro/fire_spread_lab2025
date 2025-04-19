	.file	"spread_functions.cpp"
	.text
	.section	.rodata._ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_.str1.1,"aMS",@progbits,1
.LC0:
	.string	"vector::_M_realloc_insert"
	.section	.text._ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_,"axG",@progbits,_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_,comdat
	.align 2
	.weak	_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_
	.type	_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_, @function
_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_:
.LFB9847:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdx, 16(%rsp)
	movq	8(%rdi), %rbp
	movq	(%rdi), %r15
	movq	%rbp, %rdx
	subq	%r15, %rdx
	sarq	$4, %rdx
	movabsq	$576460752303423487, %rax
	cmpq	%rax, %rdx
	je	.L17
	movq	%rdi, %r13
	movq	%rsi, %r12
	movq	%rsi, %rbx
	testq	%rdx, %rdx
	movl	$1, %eax
	cmovne	%rdx, %rax
	addq	%rdx, %rax
	jc	.L3
	movabsq	$576460752303423487, %rdx
	cmpq	%rdx, %rax
	cmovbe	%rax, %rdx
	movq	%rdx, 8(%rsp)
	movq	%rsi, %r14
	subq	%r15, %r14
	movq	$0, (%rsp)
	testq	%rax, %rax
	je	.L4
	jmp	.L10
.L17:
	leaq	.LC0(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L12:
	movq	(%rsp), %r14
	jmp	.L5
.L3:
	movq	%rsi, %r14
	subq	%r15, %r14
	movabsq	$576460752303423487, %rax
	movq	%rax, 8(%rsp)
.L10:
	movq	8(%rsp), %rdi
	salq	$4, %rdi
	call	_Znwm@PLT
	movq	%rax, (%rsp)
.L4:
	movq	16(%rsp), %rax
	vmovdqu	(%rax), %xmm2
	vmovdqa	%xmm2, 16(%rsp)
	movq	(%rsp), %rdx
	vmovdqu	%xmm2, (%rdx,%r14)
	cmpq	%r15, %r12
	je	.L12
	movq	%r12, %r14
	subq	%r15, %r14
	movq	%r15, %rax
.L6:
	vmovdqu	(%rax), %xmm0
	vmovdqu	%xmm0, (%rdx)
	addq	$16, %rax
	addq	$16, %rdx
	cmpq	%rbx, %rax
	jne	.L6
	addq	(%rsp), %r14
.L5:
	addq	$16, %r14
	cmpq	%rbp, %r12
	je	.L7
	movq	%rbp, %rdx
	subq	%r12, %rdx
	movq	%r14, %rax
.L8:
	vmovdqu	(%rbx), %xmm1
	vmovdqu	%xmm1, (%rax)
	addq	$16, %rbx
	addq	$16, %rax
	cmpq	%rbp, %rbx
	jne	.L8
	addq	%rdx, %r14
.L7:
	testq	%r15, %r15
	je	.L9
	movq	16(%r13), %rsi
	subq	%r15, %rsi
	movq	%r15, %rdi
	call	_ZdlPvm@PLT
.L9:
	movq	(%rsp), %rcx
	movq	%rcx, 0(%r13)
	movq	%r14, 8(%r13)
	movq	8(%rsp), %rax
	salq	$4, %rax
	addq	%rcx, %rax
	movq	%rax, 16(%r13)
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE9847:
	.size	_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_, .-_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_
	.section	.text._ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_,"axG",@progbits,_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_,comdat
	.align 2
	.weak	_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_
	.type	_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_, @function
_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_:
.LFB9866:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	movq	%rsi, (%rsp)
	movq	%rdx, 8(%rsp)
	movq	8(%rdi), %r12
	movq	(%rdi), %r13
	movq	%r12, %rdx
	subq	%r13, %rdx
	sarq	$3, %rdx
	movabsq	$1152921504606846975, %rax
	cmpq	%rax, %rdx
	je	.L33
	movq	%rdi, %rbx
	testq	%rdx, %rdx
	movl	$1, %eax
	cmovne	%rdx, %rax
	addq	%rdx, %rax
	jc	.L20
	movabsq	$1152921504606846975, %r14
	cmpq	%r14, %rax
	cmovbe	%rax, %r14
	movq	(%rsp), %r15
	subq	%r13, %r15
	movl	$0, %ebp
	testq	%rax, %rax
	je	.L21
	jmp	.L28
.L33:
	leaq	.LC0(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L34:
	movq	%r15, %rdx
	movq	%r13, %rsi
	movq	%rbp, %rdi
	call	memmove@PLT
	leaq	8(%rbp,%r15), %r15
	subq	(%rsp), %r12
	testq	%r12, %r12
	jg	.L23
	addq	%r12, %r15
	movq	16(%rbx), %rsi
	subq	%r13, %rsi
	jmp	.L27
.L20:
	movq	(%rsp), %r15
	subq	%r13, %r15
	movabsq	$1152921504606846975, %r14
.L28:
	leaq	0(,%r14,8), %rdi
	call	_Znwm@PLT
	movq	%rax, %rbp
.L21:
	movq	8(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, 0(%rbp,%r15)
	testq	%r15, %r15
	jg	.L34
	leaq	8(%rbp,%r15), %r15
	subq	(%rsp), %r12
	testq	%r12, %r12
	jle	.L25
.L23:
	movq	%r12, %rdx
	movq	(%rsp), %rsi
	movq	%r15, %rdi
	call	memcpy@PLT
.L25:
	addq	%r12, %r15
	testq	%r13, %r13
	je	.L26
	movq	16(%rbx), %rsi
	subq	%r13, %rsi
.L27:
	movq	%r13, %rdi
	call	_ZdlPvm@PLT
.L26:
	movq	%rbp, (%rbx)
	movq	%r15, 8(%rbx)
	leaq	0(%rbp,%r14,8), %rax
	movq	%rax, 16(%rbx)
	addq	$24, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE9866:
	.size	_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_, .-_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_
	.section	.text._ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv,"axG",@progbits,_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv,comdat
	.align 2
	.weak	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	.type	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv, @function
_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv:
.LFB9941:
	.cfi_startproc
	endbr64
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %rbx
	movq	(%rdi), %rdi
	testq	%rdi, %rdi
	je	.L35
	movq	32(%rbx), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	movq	$0, (%rbx)
	movl	$0, 8(%rbx)
	movq	$0, 16(%rbx)
	movl	$0, 24(%rbx)
	movq	$0, 32(%rbx)
.L35:
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE9941:
	.size	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv, .-_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	.section	.text._ZNSt6vectorIbSaIbEEC2ERKS1_,"axG",@progbits,_ZNSt6vectorIbSaIbEEC5ERKS1_,comdat
	.align 2
	.weak	_ZNSt6vectorIbSaIbEEC2ERKS1_
	.type	_ZNSt6vectorIbSaIbEEC2ERKS1_, @function
_ZNSt6vectorIbSaIbEEC2ERKS1_:
.LFB9654:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA9654
	endbr64
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	subq	$24, %rsp
	.cfi_def_cfa_offset 64
	movq	%rdi, %rbx
	movq	%rsi, %r13
	movq	$0, (%rdi)
	movl	$0, 8(%rdi)
	movq	$0, 16(%rdi)
	movl	$0, 24(%rdi)
	movq	$0, 32(%rdi)
	movq	16(%rsi), %rax
	subq	(%rsi), %rax
	movl	24(%rsi), %edx
	leaq	(%rdx,%rax,8), %rbp
	testq	%rbp, %rbp
	jne	.L54
.L39:
	movq	(%rbx), %r12
	movl	24(%r13), %ebp
	movq	16(%r13), %rbx
	movq	0(%r13), %rsi
	movq	%rbx, %r13
	subq	%rsi, %r13
	cmpq	%rbx, %rsi
	je	.L41
	movq	%r13, %rdx
	movq	%r12, %rdi
	call	memmove@PLT
.L41:
	leaq	(%r12,%r13), %r8
	movl	%ebp, %edi
	testq	%rdi, %rdi
	jle	.L38
	movl	$0, %esi
	movl	$0, %ecx
	movl	$1, %r9d
	jmp	.L49
.L54:
	leaq	63(%rbp), %r12
	shrq	$6, %r12
	salq	$3, %r12
	movq	%r12, %rdi
.LEHB0:
	call	_Znwm@PLT
.LEHE0:
	addq	%rax, %r12
	movq	%r12, 32(%rbx)
	movq	%rax, (%rbx)
	movl	$0, 8(%rbx)
	leaq	63(%rbp), %rdx
	testq	%rbp, %rbp
	cmovns	%rbp, %rdx
	sarq	$6, %rdx
	leaq	(%rax,%rdx,8), %rdx
	movq	%rbp, %rax
	sarq	$63, %rax
	shrq	$58, %rax
	addq	%rax, %rbp
	andl	$63, %ebp
	subq	%rax, %rbp
	movq	%rbp, %rax
	js	.L55
.L40:
	movq	%rdx, (%rsp)
	movl	%eax, 8(%rsp)
	movq	%rdx, 16(%rbx)
	movl	%eax, 24(%rbx)
	jmp	.L39
.L55:
	leaq	64(%rbp), %rax
	subq	$8, %rdx
	jmp	.L40
.L43:
	andn	(%r8), %rax, %rax
.L44:
	movq	%rax, (%r8)
	cmpl	$63, %ecx
	je	.L45
	incl	%ecx
.L46:
	cmpl	$63, %esi
	je	.L47
	incl	%esi
.L48:
	decq	%rdi
	je	.L38
.L49:
	shlx	%rsi, %r9, %rax
	shlx	%rcx, %r9, %rdx
	andq	(%rbx), %rdx
	je	.L43
	orq	(%r8), %rax
	jmp	.L44
.L45:
	addq	$8, %rbx
	movl	$0, %ecx
	jmp	.L46
.L47:
	addq	$8, %r8
	movl	$0, %esi
	jmp	.L48
.L51:
	endbr64
	movq	%rax, %rbp
	movq	%rbx, %rdi
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	movq	%rbp, %rdi
.LEHB1:
	call	_Unwind_Resume@PLT
.LEHE1:
.L38:
	addq	$24, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE9654:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table._ZNSt6vectorIbSaIbEEC2ERKS1_,"aG",@progbits,_ZNSt6vectorIbSaIbEEC5ERKS1_,comdat
.LLSDA9654:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE9654-.LLSDACSB9654
.LLSDACSB9654:
	.uleb128 .LEHB0-.LFB9654
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L51-.LFB9654
	.uleb128 0
	.uleb128 .LEHB1-.LFB9654
	.uleb128 .LEHE1-.LEHB1
	.uleb128 0
	.uleb128 0
.LLSDACSE9654:
	.section	.text._ZNSt6vectorIbSaIbEEC2ERKS1_,"axG",@progbits,_ZNSt6vectorIbSaIbEEC5ERKS1_,comdat
	.size	_ZNSt6vectorIbSaIbEEC2ERKS1_, .-_ZNSt6vectorIbSaIbEEC2ERKS1_
	.weak	_ZNSt6vectorIbSaIbEEC1ERKS1_
	.set	_ZNSt6vectorIbSaIbEEC1ERKS1_,_ZNSt6vectorIbSaIbEEC2ERKS1_
	.section	.text._ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_,"axG",@progbits,_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_,comdat
	.align 2
	.weak	_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_
	.type	_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_, @function
_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_:
.LFB10016:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdx, 16(%rsp)
	movq	8(%rdi), %rbp
	movq	(%rdi), %r15
	movq	%rbp, %rdx
	subq	%r15, %rdx
	sarq	$4, %rdx
	movabsq	$576460752303423487, %rax
	cmpq	%rax, %rdx
	je	.L72
	movq	%rdi, %r13
	movq	%rsi, %r12
	movq	%rsi, %rbx
	testq	%rdx, %rdx
	movl	$1, %eax
	cmovne	%rdx, %rax
	addq	%rdx, %rax
	jc	.L58
	movabsq	$576460752303423487, %rdx
	cmpq	%rdx, %rax
	cmovbe	%rax, %rdx
	movq	%rdx, 8(%rsp)
	movq	%rsi, %r14
	subq	%r15, %r14
	movq	$0, (%rsp)
	testq	%rax, %rax
	je	.L59
	jmp	.L65
.L72:
	leaq	.LC0(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L67:
	movq	(%rsp), %r14
	jmp	.L60
.L58:
	movq	%rsi, %r14
	subq	%r15, %r14
	movabsq	$576460752303423487, %rax
	movq	%rax, 8(%rsp)
.L65:
	movq	8(%rsp), %rdi
	salq	$4, %rdi
	call	_Znwm@PLT
	movq	%rax, (%rsp)
.L59:
	movq	16(%rsp), %rax
	vmovdqu	(%rax), %xmm2
	vmovdqa	%xmm2, 16(%rsp)
	movq	(%rsp), %rdx
	vmovdqu	%xmm2, (%rdx,%r14)
	cmpq	%r15, %r12
	je	.L67
	movq	%r12, %r14
	subq	%r15, %r14
	movq	%r15, %rax
.L61:
	vmovdqu	(%rax), %xmm0
	vmovdqu	%xmm0, (%rdx)
	addq	$16, %rax
	addq	$16, %rdx
	cmpq	%rbx, %rax
	jne	.L61
	addq	(%rsp), %r14
.L60:
	addq	$16, %r14
	cmpq	%rbp, %r12
	je	.L62
	movq	%rbp, %rdx
	subq	%r12, %rdx
	movq	%r14, %rax
.L63:
	vmovdqu	(%rbx), %xmm1
	vmovdqu	%xmm1, (%rax)
	addq	$16, %rbx
	addq	$16, %rax
	cmpq	%rbp, %rbx
	jne	.L63
	addq	%rdx, %r14
.L62:
	testq	%r15, %r15
	je	.L64
	movq	16(%r13), %rsi
	subq	%r15, %rsi
	movq	%r15, %rdi
	call	_ZdlPvm@PLT
.L64:
	movq	(%rsp), %rcx
	movq	%rcx, 0(%r13)
	movq	%r14, 8(%r13)
	movq	8(%rsp), %rax
	salq	$4, %rax
	addq	%rcx, %rax
	movq	%rax, 16(%r13)
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE10016:
	.size	_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_, .-_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_
	.text
	.globl	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff
	.type	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff, @function
_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff:
.LFB9225:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA9225
	endbr64
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	subq	$608, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	%r10, -648(%rbp)
	movq	%rdi, -656(%rbp)
	movq	%rsi, -576(%rbp)
	movq	%rdx, %r12
	vmovss	%xmm0, -620(%rbp)
	vmovss	%xmm1, -624(%rbp)
	vmovss	%xmm2, -628(%rbp)
	vmovss	%xmm3, -632(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	movq	8(%rsi), %rax
	movq	%rax, -584(%rbp)
	movq	(%rsi), %rax
	movq	%rax, -528(%rbp)
	movq	$0, -432(%rbp)
	movq	$0, -424(%rbp)
	movq	$0, -416(%rbp)
	movq	8(%rdx), %r13
	subq	(%rdx), %r13
	sarq	$4, %r13
	movq	%r13, -488(%rbp)
	je	.L74
	movl	$0, %ebx
	leaq	-432(%rbp), %r14
	jmp	.L77
.L75:
	movq	%r14, %rdi
.LEHB2:
	call	_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_
.LEHE2:
.L76:
	incq	%rbx
	cmpq	%rbx, %r13
	je	.L74
.L77:
	movq	%rbx, %rdx
	salq	$4, %rdx
	addq	(%r12), %rdx
	movq	-424(%rbp), %rsi
	cmpq	-416(%rbp), %rsi
	je	.L75
	vmovdqu	(%rdx), %xmm2
	vmovdqu	%xmm2, (%rsi)
	addq	$16, %rsi
	movq	%rsi, -424(%rbp)
	jmp	.L76
.L74:
	movq	$0, -400(%rbp)
	movq	$0, -392(%rbp)
	movq	$0, -384(%rbp)
	leaq	-488(%rbp), %rdx
	leaq	-400(%rbp), %rdi
	movl	$0, %esi
.LEHB3:
	call	_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_
.LEHE3:
	movq	-488(%rbp), %rbx
	movq	-528(%rbp), %rax
	movq	%rax, -368(%rbp)
	movq	-584(%rbp), %r13
	movq	%r13, -360(%rbp)
	imulq	%rax, %r13
	movq	$0, -352(%rbp)
	movl	$0, -344(%rbp)
	movq	$0, -336(%rbp)
	movl	$0, -328(%rbp)
	movq	$0, -320(%rbp)
	testq	%r13, %r13
	je	.L78
	leaq	63(%r13), %r14
	shrq	$6, %r14
	salq	$3, %r14
	movq	%r14, %rdi
.LEHB4:
	call	_Znwm@PLT
.LEHE4:
	movq	%rax, %rdi
	leaq	(%rax,%r14), %rax
	movq	%rax, -320(%rbp)
	movq	%rdi, -352(%rbp)
	movl	$0, -344(%rbp)
	leaq	63(%r13), %rax
	testq	%r13, %r13
	cmovns	%r13, %rax
	sarq	$6, %rax
	leaq	(%rdi,%rax,8), %rdx
	movq	%r13, %rcx
	sarq	$63, %rcx
	shrq	$58, %rcx
	leaq	0(%r13,%rcx), %rax
	andl	$63, %eax
	subq	%rcx, %rax
	movq	%rax, %rcx
	js	.L161
.L79:
	movq	%rdx, -464(%rbp)
	movl	%ecx, -456(%rbp)
	movq	%rdx, -336(%rbp)
	movl	%ecx, -328(%rbp)
	movq	%r14, %rdx
	movl	$0, %esi
	call	memset@PLT
.L78:
	movl	$0, %ecx
	movl	$1, %esi
	testq	%rbx, %rbx
	jne	.L80
	jmp	.L81
.L161:
	leaq	64(%rax), %rcx
	subq	$8, %rdx
	jmp	.L79
.L146:
	endbr64
	movq	%rax, %rbx
	leaq	-352(%rbp), %rdi
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
.L83:
	movq	-400(%rbp), %rdi
	movq	-384(%rbp), %rsi
	subq	%rdi, %rsi
	testq	%rdi, %rdi
	je	.L130
	call	_ZdlPvm@PLT
.L130:
	movq	-432(%rbp), %rdi
	movq	-416(%rbp), %rsi
	subq	%rdi, %rsi
	testq	%rdi, %rdi
	je	.L131
	call	_ZdlPvm@PLT
.L131:
	movq	%rbx, %rdi
.LEHB5:
	call	_Unwind_Resume@PLT
.LEHE5:
.L84:
	shlx	%rdx, %rsi, %rdx
	orq	%rdx, (%rdi)
	incq	%rcx
	cmpq	%rcx, %rbx
	je	.L162
.L80:
	movq	%rcx, %rdx
	salq	$4, %rdx
	addq	(%r12), %rdx
	movq	8(%rdx), %rax
	imulq	-368(%rbp), %rax
	addq	(%rdx), %rax
	leaq	63(%rax), %rdx
	cmovns	%rax, %rdx
	sarq	$6, %rdx
	movq	-352(%rbp), %rdi
	leaq	(%rdi,%rdx,8), %rdi
	cqto
	shrq	$58, %rdx
	addq	%rdx, %rax
	andl	$63, %eax
	subq	%rdx, %rax
	movq	%rax, %rdx
	jns	.L84
	leaq	64(%rax), %rdx
	subq	$8, %rdi
	jmp	.L84
.L162:
	cmpq	$-1, %rbx
	je	.L85
.L81:
	movq	$0, -608(%rbp)
	jmp	.L113
.L170:
	leaq	-128(%rbp), %rsi
	movl	$0, %ebx
	movl	$0, %eax
	leaq	4+_ZL5MOVES(%rip), %r10
	movl	$1, %r8d
	jmp	.L92
.L166:
	cmpl	-584(%rbp), %edx
	jge	.L87
	movl	%edi, (%rsi)
	movl	%edx, 32(%rsi)
	movslq	%edi, %rdi
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	imulq	-368(%rbp), %rcx
	leaq	63(%rdi,%rcx), %r9
	addq	%rdi, %rcx
	cmovns	%rcx, %r9
	sarq	$6, %r9
	movq	-352(%rbp), %r11
	leaq	(%r11,%r9,8), %r11
	movq	%rcx, %r9
	sarq	$63, %r9
	shrq	$58, %r9
	addq	%r9, %rcx
	andl	$63, %ecx
	subq	%r9, %rcx
	movq	%rcx, %r9
	js	.L163
.L133:
	shlx	%r9, %r8, %r9
	andq	(%r11), %r9
	jne	.L89
	imulq	96(%r14), %rdx
	leaq	63(%rdi,%rdx), %rcx
	addq	%rdi, %rdx
	cmovns	%rdx, %rcx
	sarq	$6, %rcx
	movq	112(%r14), %rdi
	leaq	(%rdi,%rcx,8), %rcx
	movq	%rdx, %r9
	sarq	$63, %r9
	shrq	$58, %r9
	leaq	(%rdx,%r9), %rdi
	andl	$63, %edi
	subq	%r9, %rdi
	movq	%rdi, %rdx
	js	.L164
.L90:
	shlx	%rdx, %r8, %rdx
	andq	(%rcx), %rdx
	je	.L89
	shlx	%rax, %r8, %rdx
	orq	%rdx, %rbx
.L91:
	incq	%rax
	addq	$4, %rsi
	cmpq	$8, %rax
	je	.L165
.L92:
	movl	(%r10,%rax,8), %edx
	movl	%r12d, %edi
	addl	(%r15,%rax,8), %edi
	js	.L87
	addl	%r13d, %edx
	js	.L87
	cmpl	-528(%rbp), %edi
	jl	.L166
.L87:
	movl	%r12d, (%rsi)
	movl	%r13d, 32(%rsi)
.L89:
	shlx	%rax, %r8, %rdx
	andn	%rbx, %rdx, %rbx
	jmp	.L91
.L163:
	leaq	64(%rcx), %r9
	subq	$8, %r11
	jmp	.L133
.L164:
	leaq	64(%rdi), %rdx
	subq	$8, %rcx
	jmp	.L90
.L165:
	testq	%rbx, %rbx
	je	.L109
	movq	-648(%rbp), %rax
	vmovsd	(%rax), %xmm1
	vmovsd	8(%rax), %xmm4
	vmovsd	%xmm4, -536(%rbp)
	vmovsd	16(%rax), %xmm5
	vmovsd	%xmm5, -568(%rbp)
	vmovsd	24(%rax), %xmm7
	vmovsd	%xmm7, -560(%rbp)
	vmovsd	32(%rax), %xmm4
	vmovsd	%xmm4, -600(%rbp)
	vmovsd	40(%rax), %xmm5
	vmovsd	%xmm5, -592(%rbp)
	vmovsd	48(%rax), %xmm7
	vmovsd	%xmm7, -616(%rbp)
	vmovsd	56(%rax), %xmm2
	vmovsd	%xmm2, -552(%rbp)
	vmovsd	64(%rax), %xmm3
	vmovsd	%xmm3, -544(%rbp)
	movq	-576(%rbp), %rax
	movq	16(%rax), %r12
	movq	32(%rax), %r14
	movq	152(%rax), %r13
	movq	168(%rax), %rdi
	movq	192(%rax), %r11
	movq	208(%rax), %r10
	movq	232(%rax), %r9
	movq	248(%rax), %r8
	movl	$0, %eax
	movq	%rbx, -504(%rbp)
	movq	%rdi, %rbx
.L94:
	movslq	-128(%rbp,%rax,4), %rcx
	movslq	-96(%rbp,%rax,4), %rdx
	movq	%rdx, %rsi
	imulq	%r12, %rsi
	addq	%rcx, %rsi
	movswl	(%r14,%rsi,2), %esi
	vxorps	%xmm4, %xmm4, %xmm4
	vcvtsi2ssl	%esi, %xmm4, %xmm0
	vmovss	%xmm0, -304(%rbp,%rax,4)
	movq	%rdx, %rsi
	imulq	%r13, %rsi
	leaq	(%rbx,%rcx), %rdi
	movzbl	(%rdi,%rsi), %esi
	movb	%sil, -312(%rbp,%rax)
	movq	%rdx, %rsi
	imulq	%r11, %rsi
	addq	%rcx, %rsi
	vmovss	(%r10,%rsi,4), %xmm0
	vmovss	%xmm0, -192(%rbp,%rax,4)
	imulq	%r9, %rdx
	addq	%rcx, %rdx
	vmovss	(%r8,%rdx,4), %xmm0
	vmovss	%xmm0, -160(%rbp,%rax,4)
	incq	%rax
	cmpq	$8, %rax
	jne	.L94
	movq	-504(%rbp), %rbx
	movl	$0, %r12d
	vcvtsd2ss	%xmm1, %xmm1, %xmm5
	vmovss	%xmm5, -508(%rbp)
	leaq	_ZL6ANGLES(%rip), %r14
	jmp	.L100
.L168:
	vcvtss2sd	%xmm6, %xmm6, %xmm0
	vaddsd	-600(%rbp), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm5
	vmovss	%xmm5, -504(%rbp)
.L135:
	movl	$1, %eax
	shlx	%r12, %rax, %rax
	vxorps	%xmm0, %xmm0, %xmm0
	testq	%rbx, %rax
	je	.L99
	movl	-304(%rbp,%r12,4), %r13d
	movswl	-464(%rbp), %eax
	vxorps	%xmm3, %xmm3, %xmm3
	vcvtsi2ssl	%eax, %xmm3, %xmm0
	vmovd	%r13d, %xmm3
	vsubss	%xmm0, %xmm3, %xmm0
	vdivss	-620(%rbp), %xmm0, %xmm0
	call	atanf@PLT
	call	sinf@PLT
	vmovss	%xmm0, -512(%rbp)
	vmovss	(%r14,%r12,4), %xmm0
	vsubss	-460(%rbp), %xmm0, %xmm0
	call	cosf@PLT
	vmovaps	%xmm0, %xmm1
	vxorpd	%xmm7, %xmm7, %xmm7
	vcvtss2sd	-192(%rbp,%r12,4), %xmm7, %xmm0
	vmulsd	-552(%rbp), %xmm0, %xmm0
	vcvtss2sd	-160(%rbp,%r12,4), %xmm7, %xmm2
	vmulsd	-544(%rbp), %xmm2, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm0
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vmulsd	-536(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovd	%r13d, %xmm4
	vsubss	-624(%rbp), %xmm4, %xmm1
	vdivss	-628(%rbp), %xmm1, %xmm1
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vmulsd	-568(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtss2sd	-512(%rbp), %xmm1, %xmm1
	vmulsd	-560(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtss2sd	-504(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vxorps	.LC2(%rip), %xmm0, %xmm0
	call	expf@PLT
	vaddss	.LC3(%rip), %xmm0, %xmm0
	vmovss	-632(%rbp), %xmm6
	vdivss	%xmm0, %xmm6, %xmm0
.L99:
	vmovss	%xmm0, -224(%rbp,%r12,4)
	incq	%r12
	cmpq	$8, %r12
	je	.L167
.L100:
	vmovss	-508(%rbp), %xmm6
	vmovss	%xmm6, -504(%rbp)
	movzbl	-312(%rbp,%r12), %eax
	cmpb	$2, %al
	je	.L95
	cmpb	$3, %al
	je	.L96
	cmpb	$1, %al
	je	.L168
	cmpb	$4, %al
	jne	.L135
	vxorps	%xmm0, %xmm0, %xmm0
	jmp	.L99
.L95:
	vcvtss2sd	-508(%rbp), %xmm0, %xmm0
	vaddsd	-592(%rbp), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm2
	vmovss	%xmm2, -504(%rbp)
	jmp	.L135
.L96:
	vcvtss2sd	-508(%rbp), %xmm0, %xmm0
	vaddsd	-616(%rbp), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm6
	vmovss	%xmm6, -504(%rbp)
	jmp	.L135
.L167:
	leaq	rng(%rip), %rdx
	leaq	-256(%rbp), %rsi
	leaq	32(%rdx), %rdi
	vmovss	.LC4(%rip), %xmm1
.L103:
	movl	(%rdx), %ecx
	movl	%ecx, %eax
	sall	$13, %eax
	xorl	%ecx, %eax
	movl	%eax, %ecx
	shrl	$17, %ecx
	xorl	%eax, %ecx
	movl	%ecx, %eax
	sall	$5, %eax
	xorl	%ecx, %eax
	movl	%eax, (%rdx)
	movl	%eax, %eax
	vxorps	%xmm5, %xmm5, %xmm5
	vcvtsi2ssq	%rax, %xmm5, %xmm0
	vmulss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, (%rsi)
	addq	$4, %rdx
	addq	$4, %rsi
	cmpq	%rdx, %rdi
	jne	.L103
	vmovups	-256(%rbp), %ymm6
	vcmpps	$17, -224(%rbp), %ymm6, %ymm0
	vmovmskps	%ymm0, %r13d
	testb	%r13b, %r13b
	jne	.L169
.L109:
	incq	-520(%rbp)
	movq	-520(%rbp), %rax
	cmpq	%rax, -608(%rbp)
	je	.L86
.L110:
	movq	-520(%rbp), %rax
	salq	$4, %rax
	addq	-432(%rbp), %rax
	movq	(%rax), %r12
	movq	8(%rax), %r13
	leaq	-464(%rbp), %rdi
	movq	%r12, %rdx
	movq	%r13, %rcx
	movq	-576(%rbp), %r14
	movq	%r14, %rsi
.LEHB6:
	call	_ZNK9LandscapeixESt4pairImmE@PLT
	jmp	.L170
.L169:
	movl	$0, %r12d
	movzbl	%r13b, %r13d
	jmp	.L108
.L105:
	leaq	-480(%rbp), %rdx
	leaq	-432(%rbp), %rdi
	call	_ZNSt6vectorISt4pairImmESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_
	jmp	.L106
.L107:
	movl	$1, %eax
	shlx	%rcx, %rax, %rax
	orq	%rax, (%rdx)
	incq	-640(%rbp)
.L104:
	incq	%r12
	cmpq	$8, %r12
	je	.L109
.L108:
	btl	%r12d, %r13d
	jnc	.L104
	movslq	-128(%rbp,%r12,4), %r14
	movq	%r14, -480(%rbp)
	movslq	-96(%rbp,%r12,4), %rbx
	movq	%rbx, -472(%rbp)
	movq	-424(%rbp), %rsi
	cmpq	-416(%rbp), %rsi
	je	.L105
	vmovdqa	-480(%rbp), %xmm4
	vmovdqu	%xmm4, (%rsi)
	addq	$16, %rsi
	movq	%rsi, -424(%rbp)
.L106:
	imulq	-368(%rbp), %rbx
	leaq	63(%r14,%rbx), %rax
	addq	%r14, %rbx
	cmovns	%rbx, %rax
	sarq	$6, %rax
	movq	-352(%rbp), %rdx
	leaq	(%rdx,%rax,8), %rdx
	movq	%rbx, %rax
	sarq	$63, %rax
	shrq	$58, %rax
	addq	%rax, %rbx
	andl	$63, %ebx
	subq	%rax, %rbx
	movq	%rbx, %rcx
	jns	.L107
	leaq	64(%rbx), %rcx
	subq	$8, %rdx
	jmp	.L107
.L138:
	movq	-608(%rbp), %rax
	movq	%rax, -640(%rbp)
.L86:
	movq	-640(%rbp), %rax
	movq	%rax, -488(%rbp)
	movq	-392(%rbp), %rsi
	cmpq	-384(%rbp), %rsi
	je	.L111
	movq	%rax, (%rsi)
	addq	$8, %rsi
	movq	%rsi, -392(%rbp)
.L112:
	movq	-640(%rbp), %rbx
	cmpq	%rbx, -608(%rbp)
	je	.L85
.L113:
	movq	-608(%rbp), %rbx
	movq	%rbx, -520(%rbp)
	movq	-488(%rbp), %rax
	movq	%rax, -608(%rbp)
	cmpq	%rbx, %rax
	jbe	.L138
	movq	%rax, -640(%rbp)
	leaq	_ZL5MOVES(%rip), %r15
	jmp	.L110
.L111:
	leaq	-488(%rbp), %rdx
	leaq	-400(%rbp), %rdi
	call	_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_
	jmp	.L112
.L85:
	movq	-656(%rbp), %r15
	movq	-528(%rbp), %rax
	movq	%rax, (%r15)
	movq	-584(%rbp), %rax
	movq	%rax, 8(%r15)
	movq	-368(%rbp), %rax
	movq	%rax, 16(%r15)
	movq	-360(%rbp), %rax
	movq	%rax, 24(%r15)
	leaq	-352(%rbp), %rsi
	leaq	32(%r15), %rdi
	call	_ZNSt6vectorIbSaIbEEC1ERKS1_
	movq	-424(%rbp), %r12
	movq	-432(%rbp), %rbx
	movq	$0, 72(%r15)
	movq	$0, 80(%r15)
	movq	$0, 88(%r15)
	movq	%r12, %r13
	subq	%rbx, %r13
	je	.L141
	movabsq	$9223372036854775792, %rax
	cmpq	%rax, %r13
	ja	.L171
	movq	%r13, %rdi
	call	_Znwm@PLT
	jmp	.L172
.L171:
	testq	%r13, %r13
	jns	.L116
	call	_ZSt28__throw_bad_array_new_lengthv@PLT
.L145:
	endbr64
	movq	%rax, %rbx
	leaq	-352(%rbp), %rdi
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	jmp	.L83
.L116:
	call	_ZSt17__throw_bad_allocv@PLT
.L172:
	movq	%rax, %rdx
.L114:
	movq	-656(%rbp), %rax
	movq	%rdx, 72(%rax)
	movq	%rdx, 80(%rax)
	addq	%rdx, %r13
	movq	%r13, 88(%rax)
	cmpq	%rbx, %r12
	je	.L117
	subq	%rbx, %r12
	movl	$0, %eax
.L118:
	vmovdqu	(%rbx,%rax), %xmm7
	vmovdqu	%xmm7, (%rdx,%rax)
	addq	$16, %rax
	cmpq	%r12, %rax
	jne	.L118
	addq	%r12, %rdx
.L117:
	movq	-656(%rbp), %rax
	movq	%rdx, 80(%rax)
	movq	-392(%rbp), %r15
	movq	-400(%rbp), %r12
	movq	$0, 96(%rax)
	movq	$0, 104(%rax)
	movq	$0, 112(%rax)
	movq	%r15, %r13
	subq	%r12, %r13
	je	.L142
	movabsq	$9223372036854775800, %rax
	cmpq	%rax, %r13
	ja	.L173
	movq	%r13, %rdi
	call	_Znwm@PLT
	jmp	.L174
.L141:
	movl	$0, %edx
	jmp	.L114
.L173:
	testq	%r13, %r13
	jns	.L121
	call	_ZSt28__throw_bad_array_new_lengthv@PLT
.L121:
	call	_ZSt17__throw_bad_allocv@PLT
.LEHE6:
.L174:
	movq	%rax, %rdi
.L119:
	movq	-656(%rbp), %rax
	movq	%rdi, 96(%rax)
	movq	%rdi, 104(%rax)
	leaq	(%rdi,%r13), %r14
	movq	%r14, 112(%rax)
	cmpq	%r12, %r15
	je	.L122
	movq	%r13, %rdx
	movq	%r12, %rsi
	call	memmove@PLT
	movq	-656(%rbp), %rax
	movq	%r14, 104(%rax)
	movq	-352(%rbp), %rax
	testq	%rax, %rax
	je	.L134
.L123:
	movq	-320(%rbp), %rsi
	subq	%rax, %rsi
	movq	%rax, %rdi
	call	_ZdlPvm@PLT
.L125:
	testq	%r12, %r12
	je	.L126
.L134:
	movq	-384(%rbp), %rsi
	subq	%r12, %rsi
	movq	%r12, %rdi
	call	_ZdlPvm@PLT
.L126:
	testq	%rbx, %rbx
	je	.L73
	movq	-416(%rbp), %rsi
	subq	%rbx, %rsi
	movq	%rbx, %rdi
	call	_ZdlPvm@PLT
.L73:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L175
	movq	-656(%rbp), %rax
	addq	$608, %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
.L142:
	.cfi_restore_state
	movl	$0, %edi
	jmp	.L119
.L122:
	movq	-656(%rbp), %rax
	movq	%r14, 104(%rax)
	movq	-352(%rbp), %rax
	testq	%rax, %rax
	jne	.L123
	jmp	.L125
.L144:
	endbr64
	movq	%rax, %rbx
	jmp	.L83
.L143:
	endbr64
	movq	%rax, %rbx
	jmp	.L130
.L175:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE9225:
	.section	.gcc_except_table,"a",@progbits
.LLSDA9225:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE9225-.LLSDACSB9225
.LLSDACSB9225:
	.uleb128 .LEHB2-.LFB9225
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L143-.LFB9225
	.uleb128 0
	.uleb128 .LEHB3-.LFB9225
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L144-.LFB9225
	.uleb128 0
	.uleb128 .LEHB4-.LFB9225
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L146-.LFB9225
	.uleb128 0
	.uleb128 .LEHB5-.LFB9225
	.uleb128 .LEHE5-.LEHB5
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB6-.LFB9225
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L145-.LFB9225
	.uleb128 0
.LLSDACSE9225:
	.text
	.size	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff, .-_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff
	.type	_GLOBAL__sub_I_rng, @function
_GLOBAL__sub_I_rng:
.LFB10265:
	.cfi_startproc
	endbr64
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	leaq	_ZStL8__ioinit(%rip), %rbx
	movq	%rbx, %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	leaq	__dso_handle(%rip), %rdx
	movq	%rbx, %rsi
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	call	__cxa_atexit@PLT
	leaq	_ZL5seeds(%rip), %rdx
	leaq	rng(%rip), %rax
	leaq	32(%rax), %r9
	movl	$-1831433054, %r8d
	jmp	.L178
.L177:
	movl	%esi, (%rdi)
	addq	$4, %rdx
	addq	$4, %rax
	cmpq	%r9, %rax
	je	.L182
.L178:
	movl	(%rdx), %ecx
	movq	%rax, %rdi
	movl	%ecx, (%rax)
	movl	%r8d, %esi
	testl	%ecx, %ecx
	je	.L177
	movl	%ecx, %esi
	jmp	.L177
.L182:
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE10265:
	.size	_GLOBAL__sub_I_rng, .-_GLOBAL__sub_I_rng
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_rng
	.globl	rng
	.bss
	.align 32
	.type	rng, @object
	.size	rng, 32
rng:
	.zero	32
	.section	.rodata
	.align 32
	.type	_ZL5seeds, @object
	.size	_ZL5seeds, 32
_ZL5seeds:
	.long	12345
	.long	67890
	.long	13579
	.long	24680
	.long	11223
	.long	44556
	.long	77889
	.long	99000
	.align 32
	.type	_ZL5MOVES, @object
	.size	_ZL5MOVES, 64
_ZL5MOVES:
	.long	-1
	.long	-1
	.long	-1
	.long	0
	.long	-1
	.long	1
	.long	0
	.long	-1
	.long	0
	.long	1
	.long	1
	.long	-1
	.long	1
	.long	0
	.long	1
	.long	1
	.align 32
	.type	_ZL6ANGLES, @object
	.size	_ZL6ANGLES, 32
_ZL6ANGLES:
	.long	1075235812
	.long	1078530011
	.long	1081824209
	.long	1070141403
	.long	1083624420
	.long	1061752795
	.long	0
	.long	1085271519
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC2:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC3:
	.long	1065353216
	.align 4
.LC4:
	.long	796917760
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:

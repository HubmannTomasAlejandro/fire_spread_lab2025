	.file	"spread_functions.cpp"
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.globl	_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfPbf
	.type	_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfPbf, @function
_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfPbf:
.LFB4369:
	.cfi_startproc
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
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 16(%rsp)
	vmovss	%xmm0, 32(%rsp)
	vmovss	%xmm1, 36(%rsp)
	vmovss	%xmm2, 40(%rsp)
	movq	%rdx, %r13
	movq	%rcx, %r14
	vmovss	%xmm3, 44(%rsp)
	movq	%rsi, %rbp
	movl	$0, %ebx
	jmp	.L7
.L15:
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	144(%rsp), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm5
	vmovss	%xmm5, 12(%rsp)
.L8:
	vxorps	%xmm0, %xmm0, %xmm0
	cmpb	$0, (%r14,%rbx)
	jne	.L13
.L6:
	vmovss	%xmm0, 0(%r13,%rbx,4)
	incq	%rbx
	addq	$20, %rbp
	cmpq	$8, %rbx
	je	.L14
.L7:
	vxorps	%xmm4, %xmm4, %xmm4
	vcvtsd2ss	112(%rsp), %xmm4, %xmm0
	vmovss	%xmm0, 12(%rsp)
	movq	%rbp, %r12
	movzbl	9(%rbp), %eax
	cmpb	$1, %al
	je	.L15
	cmpb	$2, %al
	je	.L16
	cmpb	$3, %al
	je	.L17
	cmpb	$4, %al
	jne	.L8
	vxorps	%xmm0, %xmm0, %xmm0
	jmp	.L6
.L16:
	vcvtss2sd	12(%rsp), %xmm0, %xmm0
	vaddsd	152(%rsp), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm3
	vmovss	%xmm3, 12(%rsp)
	jmp	.L8
.L17:
	vcvtss2sd	12(%rsp), %xmm0, %xmm0
	vaddsd	160(%rsp), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm5
	vmovss	%xmm5, 12(%rsp)
	jmp	.L8
.L13:
	movswl	(%r12), %r15d
	movq	16(%rsp), %rax
	movswl	(%rax), %edx
	movl	%r15d, %ecx
	subl	%edx, %ecx
	vxorps	%xmm6, %xmm6, %xmm6
	vcvtsi2ssl	%ecx, %xmm6, %xmm0
	vdivss	32(%rsp), %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	call	atan@PLT
	call	sin@PLT
	vmovsd	%xmm0, 24(%rsp)
	leaq	_ZL6ANGLES(%rip), %rsi
	vmovss	(%rsi,%rbx,4), %xmm0
	movq	16(%rsp), %rax
	vsubss	4(%rax), %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	call	cos@PLT
	vmovapd	%xmm0, %xmm2
	vxorpd	%xmm7, %xmm7, %xmm7
	vcvtss2sd	12(%r12), %xmm7, %xmm0
	vmulsd	168(%rsp), %xmm0, %xmm1
	vcvtss2sd	12(%rsp), %xmm0, %xmm0
	vaddsd	%xmm0, %xmm1, %xmm0
	vcvtss2sd	16(%r12), %xmm7, %xmm1
	vmulsd	176(%rsp), %xmm1, %xmm1
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm1, %xmm1
	vxorps	%xmm6, %xmm6, %xmm6
	vcvtsi2ssl	%r15d, %xmm6, %xmm0
	vsubss	36(%rsp), %xmm0, %xmm0
	vdivss	40(%rsp), %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vmulsd	128(%rsp), %xmm0, %xmm0
	vcvtsd2ss	%xmm2, %xmm2, %xmm2
	vcvtss2sd	%xmm2, %xmm2, %xmm2
	vmulsd	120(%rsp), %xmm2, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm0
	vcvtsd2ss	24(%rsp), %xmm2, %xmm2
	vcvtss2sd	%xmm2, %xmm2, %xmm2
	vmulsd	136(%rsp), %xmm2, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm0
	vcvtsd2ss	%xmm1, %xmm1, %xmm1
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vxorps	.LC1(%rip), %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	call	exp@PLT
	vcvtss2sd	44(%rsp), %xmm1, %xmm1
	vaddsd	.LC2(%rip), %xmm0, %xmm0
	vdivsd	%xmm0, %xmm1, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	jmp	.L6
.L14:
	addq	$56, %rsp
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
.LFE4369:
	.size	_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfPbf, .-_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfPbf
	.section	.rodata._ZNSt6vectorImSaImEE17_M_realloc_appendIJRKmEEEvDpOT_.str1.1,"aMS",@progbits,1
.LC3:
	.string	"vector::_M_realloc_append"
	.section	.text._ZNSt6vectorImSaImEE17_M_realloc_appendIJRKmEEEvDpOT_,"axG",@progbits,_ZNSt6vectorImSaImEE17_M_realloc_appendIJRKmEEEvDpOT_,comdat
	.align 2
	.weak	_ZNSt6vectorImSaImEE17_M_realloc_appendIJRKmEEEvDpOT_
	.type	_ZNSt6vectorImSaImEE17_M_realloc_appendIJRKmEEEvDpOT_, @function
_ZNSt6vectorImSaImEE17_M_realloc_appendIJRKmEEEvDpOT_:
.LFB5033:
	.cfi_startproc
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
	subq	$8, %rsp
	.cfi_def_cfa_offset 64
	movq	(%rdi), %r14
	movq	8(%rdi), %r12
	subq	%r14, %r12
	movq	%r12, %rcx
	sarq	$3, %rcx
	movabsq	$1152921504606846975, %rax
	cmpq	%rax, %rcx
	je	.L26
	movq	%rdi, %rbp
	movq	%rsi, %r13
	testq	%rcx, %rcx
	movl	$1, %eax
	cmovne	%rcx, %rax
	addq	%rcx, %rax
	movabsq	$1152921504606846975, %rdx
	cmpq	%rdx, %rax
	cmovbe	%rax, %rdx
	movq	%rdx, %r15
	cmpq	%rcx, %rax
	movabsq	$1152921504606846975, %rax
	cmovb	%rax, %r15
	salq	$3, %r15
	movq	%r15, %rdi
	call	_Znwm@PLT
	movq	%rax, %rbx
	movq	0(%r13), %rax
	movq	%rax, (%rbx,%r12)
	testq	%r12, %r12
	jg	.L27
	leaq	8(%rbx,%r12), %r12
	testq	%r14, %r14
	je	.L23
	movq	16(%rbp), %rsi
	subq	%r14, %rsi
	jmp	.L22
.L26:
	leaq	.LC3(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L27:
	movq	%r12, %rdx
	movq	%r14, %rsi
	movq	%rbx, %rdi
	call	memcpy@PLT
	leaq	8(%rbx,%r12), %r12
	movq	16(%rbp), %rsi
	subq	%r14, %rsi
.L22:
	movq	%r14, %rdi
	call	_ZdlPvm@PLT
.L23:
	movq	%rbx, 0(%rbp)
	movq	%r12, 8(%rbp)
	addq	%r15, %rbx
	movq	%rbx, 16(%rbp)
	addq	$8, %rsp
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
.LFE5033:
	.size	_ZNSt6vectorImSaImEE17_M_realloc_appendIJRKmEEEvDpOT_, .-_ZNSt6vectorImSaImEE17_M_realloc_appendIJRKmEEEvDpOT_
	.section	.text._ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv,"axG",@progbits,_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv,comdat
	.align 2
	.weak	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	.type	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv, @function
_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv:
.LFB5133:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %rbx
	movq	(%rdi), %rdi
	testq	%rdi, %rdi
	je	.L28
	movq	32(%rbx), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	movq	$0, (%rbx)
	movl	$0, 8(%rbx)
	movq	$0, 16(%rbx)
	movl	$0, 24(%rbx)
	movq	$0, 32(%rbx)
.L28:
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE5133:
	.size	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv, .-_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	.section	.text._ZNSt6vectorIbSaIbEEC2ERKS1_,"axG",@progbits,_ZNSt6vectorIbSaIbEEC5ERKS1_,comdat
	.align 2
	.weak	_ZNSt6vectorIbSaIbEEC2ERKS1_
	.type	_ZNSt6vectorIbSaIbEEC2ERKS1_, @function
_ZNSt6vectorIbSaIbEEC2ERKS1_:
.LFB4796:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA4796
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
	movq	$0, (%rdi)
	movl	$0, 8(%rdi)
	movq	$0, 16(%rdi)
	movl	$0, 24(%rdi)
	movq	$0, 32(%rdi)
	movq	(%rsi), %r15
	movq	16(%rsi), %rbp
	movq	%rbp, %r14
	subq	%r15, %r14
	movl	24(%rsi), %ebx
	leaq	(%rbx,%r14,8), %r13
	movq	$0, (%rsp)
	testq	%r13, %r13
	je	.L32
	movq	%rdi, %r12
	leaq	63(%r13), %rax
	shrq	$6, %rax
	salq	$3, %rax
	movq	%rax, 8(%rsp)
	movq	%rax, %rdi
.LEHB0:
	call	_Znwm@PLT
.LEHE0:
	movq	%rax, %rcx
	movq	%rax, (%rsp)
	movq	8(%rsp), %rax
	addq	%rcx, %rax
	movq	%rax, 32(%r12)
	movq	%rcx, (%r12)
	movl	$0, 8(%r12)
	leaq	63(%r13), %rax
	testq	%r13, %r13
	cmovns	%r13, %rax
	sarq	$6, %rax
	leaq	(%rcx,%rax,8), %rax
	movq	%r13, %rdx
	sarq	$63, %rdx
	shrq	$58, %rdx
	addq	%rdx, %r13
	andl	$63, %r13d
	subq	%rdx, %r13
	movq	%r13, %rdx
	js	.L48
.L33:
	movq	%rax, 16(%rsp)
	movl	%edx, 24(%rsp)
	movq	%rax, 16(%r12)
	movl	%edx, 24(%r12)
	cmpq	$8, %r14
	jle	.L34
	movq	%r14, %rdx
	movq	%r15, %rsi
	movq	(%rsp), %rdi
	call	memmove@PLT
.L32:
	movq	(%rsp), %rdi
	addq	%r14, %rdi
	testq	%rbx, %rbx
	jle	.L31
	movl	$0, %esi
	movl	$0, %ecx
	movl	$1, %r8d
	jmp	.L42
.L48:
	leaq	64(%r13), %rdx
	subq	$8, %rax
	jmp	.L33
.L34:
	jne	.L32
	movq	(%r15), %rax
	movq	(%rsp), %rdi
	movq	%rax, (%rdi)
	jmp	.L32
.L36:
	andn	(%rdi), %rax, %rax
.L37:
	movq	%rax, (%rdi)
	cmpl	$63, %ecx
	je	.L38
	incl	%ecx
.L39:
	cmpl	$63, %esi
	je	.L40
	incl	%esi
.L41:
	decq	%rbx
	je	.L31
.L42:
	shlx	%rsi, %r8, %rax
	shlx	%rcx, %r8, %rdx
	andq	0(%rbp), %rdx
	je	.L36
	orq	(%rdi), %rax
	jmp	.L37
.L38:
	addq	$8, %rbp
	movl	$0, %ecx
	jmp	.L39
.L40:
	addq	$8, %rdi
	movl	$0, %esi
	jmp	.L41
.L45:
	movq	%rax, %rbx
	movq	%r12, %rdi
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	movq	%rbx, %rdi
.LEHB1:
	call	_Unwind_Resume@PLT
.LEHE1:
.L31:
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
.LFE4796:
	.section	.gcc_except_table._ZNSt6vectorIbSaIbEEC2ERKS1_,"aG",@progbits,_ZNSt6vectorIbSaIbEEC5ERKS1_,comdat
.LLSDA4796:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE4796-.LLSDACSB4796
.LLSDACSB4796:
	.uleb128 .LEHB0-.LFB4796
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L45-.LFB4796
	.uleb128 0
	.uleb128 .LEHB1-.LFB4796
	.uleb128 .LEHE1-.LEHB1
	.uleb128 0
	.uleb128 0
.LLSDACSE4796:
	.section	.text._ZNSt6vectorIbSaIbEEC2ERKS1_,"axG",@progbits,_ZNSt6vectorIbSaIbEEC5ERKS1_,comdat
	.size	_ZNSt6vectorIbSaIbEEC2ERKS1_, .-_ZNSt6vectorIbSaIbEEC2ERKS1_
	.weak	_ZNSt6vectorIbSaIbEEC1ERKS1_
	.set	_ZNSt6vectorIbSaIbEEC1ERKS1_,_ZNSt6vectorIbSaIbEEC2ERKS1_
	.text
	.globl	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff
	.type	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff, @function
_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff:
.LFB4370:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA4370
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	pushq	%rbx
	subq	$640, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 10, -56
	.cfi_offset 3, -64
	leaq	16(%rbp), %r10
	movq	%r10, -656(%rbp)
	movq	%rdi, -688(%rbp)
	movq	%rsi, -608(%rbp)
	movq	%rdx, %r15
	vmovss	%xmm0, -636(%rbp)
	vmovss	%xmm1, -640(%rbp)
	vmovss	%xmm2, -644(%rbp)
	vmovss	%xmm3, -648(%rbp)
	movq	8(%rsi), %rax
	movq	%rax, -616(%rbp)
	movq	(%rsi), %rax
	movq	%rax, -568(%rbp)
	movq	8(%rdx), %rax
	movq	(%rdx), %rdx
	movq	%rax, %rcx
	subq	%rdx, %rcx
	sarq	$4, %rcx
	movq	%rcx, %r14
	movq	%rcx, -136(%rbp)
	cmpq	%rax, %rdx
	je	.L117
	movq	$0, -600(%rbp)
	movq	$0, -584(%rbp)
	movq	$0, -592(%rbp)
	movl	$0, %ebx
	jmp	.L58
.L51:
	movq	-584(%rbp), %rax
	movq	-592(%rbp), %rsi
	subq	%rsi, %rax
	movq	%rax, -544(%rbp)
	movq	%rax, %rcx
	sarq	$4, %rcx
	movabsq	$576460752303423487, %rax
	cmpq	%rax, %rcx
	je	.L149
	testq	%rcx, %rcx
	movl	$1, %eax
	cmovne	%rcx, %rax
	addq	%rcx, %rax
	movabsq	$576460752303423487, %rdx
	cmpq	%rdx, %rax
	cmovbe	%rax, %rdx
	cmpq	%rcx, %rax
	movabsq	$576460752303423487, %rax
	cmovnb	%rdx, %rax
	salq	$4, %rax
	movq	%rax, -576(%rbp)
	movq	%rax, %rdi
.LEHB2:
	call	_Znwm@PLT
	jmp	.L150
.L149:
	leaq	.LC3(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE2:
.L125:
	movq	%rax, %rbx
	jmp	.L111
.L150:
	movq	%rax, %r13
	vmovdqu	(%r12), %xmm3
	vmovdqa	%xmm3, -560(%rbp)
	movq	-544(%rbp), %rax
	vmovdqu	%xmm3, 0(%r13,%rax)
	movq	-584(%rbp), %rcx
	movq	-592(%rbp), %rax
	cmpq	%rax, %rcx
	je	.L119
	movq	%r13, %rdx
	.p2align 5
.L56:
	vmovdqu	(%rax), %xmm7
	vmovdqu	%xmm7, (%rdx)
	addq	$16, %rax
	addq	$16, %rdx
	cmpq	%rax, %rcx
	jne	.L56
	movq	-584(%rbp), %rax
	movq	-592(%rbp), %rsi
	subq	%rsi, %rax
	addq	%r13, %rax
.L55:
	addq	$16, %rax
	movq	%rax, -584(%rbp)
	movq	-592(%rbp), %rdi
	testq	%rdi, %rdi
	je	.L57
	movq	-600(%rbp), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L57:
	movq	-576(%rbp), %rax
	addq	%r13, %rax
	movq	%rax, -600(%rbp)
	movq	%r13, -592(%rbp)
.L52:
	incq	%rbx
	cmpq	%r14, %rbx
	jnb	.L50
.L58:
	movq	%rbx, %r12
	salq	$4, %r12
	addq	(%r15), %r12
	movq	-584(%rbp), %rax
	movq	-600(%rbp), %rdi
	cmpq	%rdi, %rax
	je	.L51
	vmovdqu	(%r12), %xmm7
	vmovdqa	%xmm7, -544(%rbp)
	vmovdqu	%xmm7, (%rax)
	addq	$16, %rax
	movq	%rax, -584(%rbp)
	jmp	.L52
.L119:
	movq	%r13, %rax
	jmp	.L55
.L117:
	movq	$0, -600(%rbp)
	movq	$0, -584(%rbp)
	movq	$0, -592(%rbp)
.L50:
	movq	$0, -160(%rbp)
	movq	$0, -152(%rbp)
	movq	$0, -144(%rbp)
	leaq	-136(%rbp), %rsi
	leaq	-160(%rbp), %rdi
.LEHB3:
	call	_ZNSt6vectorImSaImEE17_M_realloc_appendIJRKmEEEvDpOT_
.LEHE3:
	movq	-160(%rbp), %rax
	movq	%rax, -680(%rbp)
	movq	-152(%rbp), %rax
	movq	%rax, -664(%rbp)
	movq	-144(%rbp), %rax
	movq	%rax, -672(%rbp)
	movq	-568(%rbp), %rbx
	movq	%rbx, -224(%rbp)
	movq	-616(%rbp), %rax
	movq	%rax, -216(%rbp)
	imulq	%rax, %rbx
	movq	$0, -208(%rbp)
	movl	$0, -200(%rbp)
	movq	$0, -192(%rbp)
	movl	$0, -184(%rbp)
	movq	$0, -176(%rbp)
	testq	%rbx, %rbx
	je	.L59
	leaq	63(%rbx), %rax
	shrq	$6, %rax
	leaq	0(,%rax,8), %r12
	movq	%r12, %rdi
.LEHB4:
	call	_Znwm@PLT
.LEHE4:
	movq	%rax, %rdi
	leaq	(%rax,%r12), %rax
	movq	%rax, -176(%rbp)
	movq	%rdi, -208(%rbp)
	movl	$0, -200(%rbp)
	leaq	63(%rbx), %rax
	testq	%rbx, %rbx
	cmovns	%rbx, %rax
	sarq	$6, %rax
	leaq	(%rdi,%rax,8), %rdx
	movq	%rbx, %rcx
	sarq	$63, %rcx
	shrq	$58, %rcx
	leaq	(%rbx,%rcx), %rax
	andl	$63, %eax
	subq	%rcx, %rax
	movq	%rax, %rcx
	js	.L151
.L60:
	movq	%rdx, -64(%rbp)
	movl	%ecx, -56(%rbp)
	movq	%rdx, -192(%rbp)
	movl	%ecx, -184(%rbp)
	movq	%r12, %rdx
	movl	$0, %esi
	call	memset@PLT
.L59:
	movl	$0, %ecx
	movl	$1, %esi
	testq	%r14, %r14
	jne	.L64
	jmp	.L66
.L151:
	leaq	64(%rax), %rcx
	subq	$8, %rdx
	jmp	.L60
.L129:
	movq	%rax, %rbx
	leaq	-208(%rbp), %rdi
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
.L62:
	movq	-672(%rbp), %rsi
	movq	-680(%rbp), %rax
	subq	%rax, %rsi
	testq	%rax, %rax
	je	.L111
	movq	%rax, %rdi
	call	_ZdlPvm@PLT
.L111:
	movq	-600(%rbp), %rsi
	movq	-592(%rbp), %rax
	subq	%rax, %rsi
	testq	%rax, %rax
	je	.L112
	movq	%rax, %rdi
	call	_ZdlPvm@PLT
.L112:
	movq	%rbx, %rdi
.LEHB5:
	call	_Unwind_Resume@PLT
.LEHE5:
.L153:
	leaq	64(%rax), %rdx
	subq	$8, %rdi
.L63:
	shlx	%rdx, %rsi, %rdx
	orq	%rdx, (%rdi)
	incq	%rcx
	cmpq	%rcx, %r14
	je	.L152
.L64:
	movq	%rcx, %rdx
	salq	$4, %rdx
	addq	(%r15), %rdx
	movq	8(%rdx), %rax
	imulq	-224(%rbp), %rax
	addq	(%rdx), %rax
	leaq	63(%rax), %rdx
	cmovns	%rax, %rdx
	sarq	$6, %rdx
	movq	-208(%rbp), %rdi
	leaq	(%rdi,%rdx,8), %rdi
	cqto
	shrq	$58, %rdx
	addq	%rdx, %rax
	andl	$63, %eax
	subq	%rdx, %rax
	movq	%rax, %rdx
	jns	.L63
	jmp	.L153
.L152:
	cmpq	$-1, %r14
	je	.L67
.L66:
	movq	$0, -632(%rbp)
	jmp	.L93
.L157:
	leaq	-416(%rbp), %r15
	movl	$0, %r12d
	jmp	.L74
.L70:
	movq	_ZL18out_of_bounds_cell(%rip), %rcx
	movq	8+_ZL18out_of_bounds_cell(%rip), %rbx
	movq	%rcx, -128(%rbp)
	movq	%rbx, -120(%rbp)
	movl	16+_ZL18out_of_bounds_cell(%rip), %eax
	movl	%eax, -112(%rbp)
	movq	%rcx, (%r15)
	movq	%rbx, 8(%r15)
	movl	%eax, 16(%r15)
	movl	$0, %r13d
.L72:
	movb	%r13b, -464(%rbp,%r12)
	incq	%r12
	addq	$20, %r15
	cmpq	$8, %r12
	je	.L154
.L74:
	leaq	_ZL5MOVES(%rip), %rax
	movl	-544(%rbp), %r14d
	addl	(%rax,%r12,8), %r14d
	movl	%r14d, -528(%rbp,%r12,4)
	leaq	4(%rax), %rax
	movl	-560(%rbp), %ebx
	addl	(%rax,%r12,8), %ebx
	movl	%ebx, -496(%rbp,%r12,4)
	movl	%r14d, %eax
	orl	%ebx, %eax
	shrl	$31, %eax
	movl	-568(%rbp), %edi
	cmpl	%edi, %r14d
	setge	%r13b
	orb	%al, %r13b
	jne	.L70
	movl	-616(%rbp), %eax
	cmpl	%eax, %ebx
	jge	.L70
	movslq	%r14d, %r14
	movslq	%ebx, %rbx
	leaq	-128(%rbp), %rdi
	movq	%rbx, %rcx
	movq	%r14, %rdx
	movq	-608(%rbp), %rsi
.LEHB6:
	call	_ZNK9LandscapeixESt4pairImmE@PLT
	movq	%r15, %rdx
	vmovdqa	-128(%rbp), %xmm5
	vmovdqu	%xmm5, (%r15)
	movl	-112(%rbp), %eax
	movl	%eax, 16(%r15)
	imulq	-224(%rbp), %rbx
	leaq	63(%r14,%rbx), %rax
	addq	%r14, %rbx
	cmovns	%rbx, %rax
	sarq	$6, %rax
	movq	-208(%rbp), %rcx
	leaq	(%rcx,%rax,8), %rcx
	movq	%rbx, %rax
	sarq	$63, %rax
	shrq	$58, %rax
	addq	%rax, %rbx
	andl	$63, %ebx
	subq	%rax, %rbx
	movq	%rbx, %rax
	js	.L155
.L73:
	movl	$1, %edi
	shlx	%rax, %rdi, %rax
	andq	(%rcx), %rax
	jne	.L72
	movzbl	8(%rdx), %r13d
	jmp	.L72
.L155:
	leaq	64(%rbx), %rax
	subq	$8, %rcx
	jmp	.L73
.L154:
	leaq	-464(%rbp), %rcx
	leaq	-448(%rbp), %rdx
	leaq	-416(%rbp), %rsi
	leaq	-96(%rbp), %rdi
	subq	$80, %rsp
	movq	-656(%rbp), %rax
	vmovdqu	(%rax), %ymm0
	vmovdqu	%ymm0, (%rsp)
	vmovdqu	32(%rax), %ymm0
	vmovdqu	%ymm0, 32(%rsp)
	movq	64(%rax), %rax
	movq	%rax, 64(%rsp)
	vmovss	-648(%rbp), %xmm3
	vmovss	-644(%rbp), %xmm2
	vmovss	-640(%rbp), %xmm1
	vmovss	-636(%rbp), %xmm0
	call	_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfPbf
	leaq	rng(%rip), %rdx
	leaq	-256(%rbp), %rsi
	leaq	32(%rdx), %rdi
	addq	$80, %rsp
	.p2align 6
.L77:
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
	vxorps	%xmm4, %xmm4, %xmm4
	vcvtsi2ssq	%rax, %xmm4, %xmm0
	vmulss	.LC4(%rip), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi)
	addq	$4, %rdx
	addq	$4, %rsi
	cmpq	%rdx, %rdi
	jne	.L77
	movl	$0, %eax
	movl	$0, %ecx
	leaq	-256(%rbp), %rsi
	.p2align 6
.L78:
	vmovss	-448(%rbp,%rax,4), %xmm0
	vcomiss	(%rsi,%rax,4), %xmm0
	seta	%dl
	movb	%dl, -456(%rbp,%rax)
	orl	%edx, %ecx
	incq	%rax
	cmpq	$8, %rax
	jne	.L78
	testb	%cl, %cl
	jne	.L156
.L79:
	incq	-576(%rbp)
	movq	-576(%rbp), %rax
	cmpq	%rax, -632(%rbp)
	je	.L68
.L90:
	movq	-576(%rbp), %rax
	salq	$4, %rax
	movq	-592(%rbp), %rsi
	addq	%rsi, %rax
	movq	(%rax), %rsi
	movq	%rsi, -544(%rbp)
	movq	8(%rax), %rax
	movq	%rax, -560(%rbp)
	movl	$0, %eax
	.p2align 5
.L69:
	movb	$0, -464(%rbp,%rax)
	movb	$0, -456(%rbp,%rax)
	incq	%rax
	cmpq	$8, %rax
	jne	.L69
	movq	-544(%rbp), %rdx
	movq	-560(%rbp), %rcx
	leaq	-96(%rbp), %rdi
	movq	-608(%rbp), %rsi
	call	_ZNK9LandscapeixESt4pairImmE@PLT
	jmp	.L157
.L156:
	movl	$0, %r12d
	leaq	-456(%rbp), %r14
	jmp	.L89
.L81:
	movq	-584(%rbp), %rax
	movq	-592(%rbp), %rsi
	subq	%rsi, %rax
	movq	%rax, -544(%rbp)
	movq	%rax, %rcx
	sarq	$4, %rcx
	movabsq	$576460752303423487, %rax
	cmpq	%rax, %rcx
	je	.L158
	testq	%rcx, %rcx
	movl	$1, %eax
	cmovne	%rcx, %rax
	addq	%rcx, %rax
	movabsq	$576460752303423487, %rdx
	cmpq	%rdx, %rax
	cmovbe	%rax, %rdx
	cmpq	%rcx, %rax
	movabsq	$576460752303423487, %rax
	cmovb	%rax, %rdx
	salq	$4, %rdx
	movq	%rdx, %rdi
	movq	%rdx, -560(%rbp)
	call	_Znwm@PLT
	jmp	.L159
.L158:
	leaq	.LC3(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE6:
.L126:
	movq	%rax, %rbx
	jmp	.L107
.L159:
	movq	%rax, %r15
	movq	-544(%rbp), %rax
	addq	%r15, %rax
	movq	%r13, (%rax)
	movq	%rbx, 8(%rax)
	movq	-584(%rbp), %rcx
	movq	-592(%rbp), %rax
	cmpq	%rax, %rcx
	je	.L123
	movq	%r15, %rdx
	.p2align 5
.L86:
	vmovdqu	(%rax), %xmm6
	vmovdqu	%xmm6, (%rdx)
	addq	$16, %rax
	addq	$16, %rdx
	cmpq	%rax, %rcx
	jne	.L86
	movq	-584(%rbp), %rax
	movq	-592(%rbp), %rsi
	subq	%rsi, %rax
	addq	%r15, %rax
.L85:
	addq	$16, %rax
	movq	%rax, -584(%rbp)
	movq	-592(%rbp), %rdi
	testq	%rdi, %rdi
	je	.L87
	movq	-600(%rbp), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L87:
	movq	-560(%rbp), %rax
	addq	%r15, %rax
	movq	%rax, -600(%rbp)
	movq	%r15, -592(%rbp)
	jmp	.L82
.L123:
	movq	%r15, %rax
	jmp	.L85
.L88:
	movl	$1, %eax
	shlx	%rcx, %rax, %rax
	orq	%rax, (%rdx)
	incq	-624(%rbp)
.L80:
	incq	%r12
	cmpq	$8, %r12
	je	.L79
.L89:
	cmpb	$0, (%r12,%r14)
	je	.L80
	movslq	-528(%rbp,%r12,4), %r13
	movslq	-496(%rbp,%r12,4), %rbx
	movq	-584(%rbp), %rax
	movq	-600(%rbp), %rsi
	cmpq	%rsi, %rax
	je	.L81
	movq	%r13, (%rax)
	movq	%rbx, 8(%rax)
	addq	$16, %rax
	movq	%rax, -584(%rbp)
.L82:
	imulq	-224(%rbp), %rbx
	leaq	63(%r13,%rbx), %rax
	addq	%r13, %rbx
	cmovns	%rbx, %rax
	sarq	$6, %rax
	movq	-208(%rbp), %rdx
	leaq	(%rdx,%rax,8), %rdx
	movq	%rbx, %rax
	sarq	$63, %rax
	shrq	$58, %rax
	addq	%rax, %rbx
	andl	$63, %ebx
	subq	%rax, %rbx
	movq	%rbx, %rcx
	jns	.L88
	leaq	64(%rbx), %rcx
	subq	$8, %rdx
	jmp	.L88
.L121:
	movq	-632(%rbp), %rax
	movq	%rax, -624(%rbp)
.L68:
	movq	-624(%rbp), %rdi
	movq	%rdi, -136(%rbp)
	movq	-664(%rbp), %rax
	movq	-672(%rbp), %rsi
	cmpq	%rsi, %rax
	je	.L91
	movq	%rdi, (%rax)
	addq	$8, %rax
	movq	%rax, -664(%rbp)
.L92:
	movq	-624(%rbp), %rdi
	cmpq	%rdi, -632(%rbp)
	je	.L67
.L93:
	movq	-632(%rbp), %rsi
	movq	%rsi, -576(%rbp)
	movq	-136(%rbp), %rax
	movq	%rax, -632(%rbp)
	cmpq	%rax, %rsi
	jnb	.L121
	movq	%rax, -624(%rbp)
	jmp	.L90
.L91:
	movq	-680(%rbp), %rax
	movq	%rax, -160(%rbp)
	movq	-664(%rbp), %rax
	movq	%rax, -152(%rbp)
	movq	-672(%rbp), %rax
	movq	%rax, -144(%rbp)
	leaq	-136(%rbp), %rsi
	leaq	-160(%rbp), %rdi
.LEHB7:
	call	_ZNSt6vectorImSaImEE17_M_realloc_appendIJRKmEEEvDpOT_
.LEHE7:
	movq	-160(%rbp), %rax
	movq	%rax, -680(%rbp)
	movq	-152(%rbp), %rax
	movq	%rax, -664(%rbp)
	movq	-144(%rbp), %rax
	movq	%rax, -672(%rbp)
	jmp	.L92
.L67:
	movq	-688(%rbp), %rbx
	movq	-568(%rbp), %rax
	movq	%rax, (%rbx)
	movq	-616(%rbp), %rax
	movq	%rax, 8(%rbx)
	movq	-224(%rbp), %rax
	movq	%rax, 16(%rbx)
	movq	-216(%rbp), %rax
	movq	%rax, 24(%rbx)
	leaq	32(%rbx), %r12
	leaq	-208(%rbp), %rsi
	movq	%r12, %rdi
.LEHB8:
	call	_ZNSt6vectorIbSaIbEEC1ERKS1_
.LEHE8:
	movq	$0, 72(%rbx)
	movq	$0, 80(%rbx)
	movq	$0, 88(%rbx)
	movq	-584(%rbp), %rbx
	movq	-592(%rbp), %rax
	subq	%rax, %rbx
	je	.L124
	movabsq	$9223372036854775792, %rax
	cmpq	%rbx, %rax
	jb	.L160
	movq	%rbx, %rdi
.LEHB9:
	call	_Znwm@PLT
	jmp	.L161
.L160:
	call	_ZSt28__throw_bad_array_new_lengthv@PLT
.LEHE9:
.L127:
	movq	%rax, %rbx
	jmp	.L106
.L161:
	movq	%rax, %rcx
.L94:
	movq	-688(%rbp), %rax
	movq	%rcx, 72(%rax)
	movq	%rcx, 80(%rax)
	addq	%rcx, %rbx
	movq	%rbx, 88(%rax)
	movq	-592(%rbp), %rsi
	movq	-584(%rbp), %rax
	cmpq	%rax, %rsi
	je	.L96
	subq	%rsi, %rax
	movl	$0, %edx
	.p2align 5
.L97:
	vmovdqu	(%rsi,%rdx), %xmm2
	vmovdqu	%xmm2, (%rcx,%rdx)
	addq	$16, %rdx
	cmpq	%rdx, %rax
	jne	.L97
	addq	%rax, %rcx
.L96:
	movq	-688(%rbp), %rax
	movq	%rcx, 80(%rax)
	movq	$0, 96(%rax)
	movq	$0, 104(%rax)
	movq	$0, 112(%rax)
	movq	-664(%rbp), %rbx
	movq	-680(%rbp), %rax
	subq	%rax, %rbx
	je	.L98
	movabsq	$9223372036854775800, %rax
	cmpq	%rbx, %rax
	jb	.L162
	movq	%rbx, %rdi
.LEHB10:
	call	_Znwm@PLT
	jmp	.L163
.L124:
	movl	$0, %ecx
	jmp	.L94
.L162:
	call	_ZSt28__throw_bad_array_new_lengthv@PLT
.LEHE10:
.L128:
	movq	%rax, %rbx
	movq	-688(%rbp), %rax
	movq	72(%rax), %rdi
	movq	88(%rax), %rsi
	subq	%rdi, %rsi
	testq	%rdi, %rdi
	je	.L106
	call	_ZdlPvm@PLT
.L106:
	movq	%r12, %rdi
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
.L107:
	leaq	-208(%rbp), %rdi
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	jmp	.L62
.L163:
	movq	%rax, %rdi
	movq	-688(%rbp), %rax
	movq	%rdi, 96(%rax)
	movq	%rdi, 104(%rax)
	leaq	(%rdi,%rbx), %r12
	movq	%r12, 112(%rax)
	cmpq	$8, %rbx
	jle	.L100
	movq	%rbx, %rdx
	movq	-680(%rbp), %rsi
	call	memmove@PLT
.L101:
	movq	-688(%rbp), %rax
	movq	%r12, 104(%rax)
	movq	-208(%rbp), %rdi
	testq	%rdi, %rdi
	je	.L115
.L113:
	movq	-176(%rbp), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L114:
	cmpq	$0, -680(%rbp)
	je	.L102
.L115:
	movq	-672(%rbp), %rsi
	movq	-680(%rbp), %rdi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L102:
	movq	-592(%rbp), %rdi
	testq	%rdi, %rdi
	je	.L49
	movq	-600(%rbp), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L49:
	movq	-688(%rbp), %rax
	leaq	-48(%rbp), %rsp
	popq	%rbx
	popq	%r10
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L100:
	.cfi_restore_state
	movq	-680(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, (%rdi)
	jmp	.L101
.L131:
	movq	%rax, %rbx
	movq	-160(%rbp), %rax
	movq	%rax, -680(%rbp)
	movq	-144(%rbp), %rax
	movq	%rax, -672(%rbp)
	jmp	.L107
.L130:
	movq	%rax, %rbx
	movq	-160(%rbp), %rax
	movq	%rax, -680(%rbp)
	movq	-144(%rbp), %rax
	movq	%rax, -672(%rbp)
	jmp	.L62
.L98:
	movq	-688(%rbp), %rax
	movq	$0, 96(%rax)
	movq	$0, 112(%rax)
	movq	$0, 104(%rax)
	movq	-208(%rbp), %rdi
	testq	%rdi, %rdi
	jne	.L113
	jmp	.L114
	.cfi_endproc
.LFE4370:
	.section	.gcc_except_table,"a",@progbits
.LLSDA4370:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE4370-.LLSDACSB4370
.LLSDACSB4370:
	.uleb128 .LEHB2-.LFB4370
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L125-.LFB4370
	.uleb128 0
	.uleb128 .LEHB3-.LFB4370
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L130-.LFB4370
	.uleb128 0
	.uleb128 .LEHB4-.LFB4370
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L129-.LFB4370
	.uleb128 0
	.uleb128 .LEHB5-.LFB4370
	.uleb128 .LEHE5-.LEHB5
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB6-.LFB4370
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L126-.LFB4370
	.uleb128 0
	.uleb128 .LEHB7-.LFB4370
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L131-.LFB4370
	.uleb128 0
	.uleb128 .LEHB8-.LFB4370
	.uleb128 .LEHE8-.LEHB8
	.uleb128 .L126-.LFB4370
	.uleb128 0
	.uleb128 .LEHB9-.LFB4370
	.uleb128 .LEHE9-.LEHB9
	.uleb128 .L127-.LFB4370
	.uleb128 0
	.uleb128 .LEHB10-.LFB4370
	.uleb128 .LEHE10-.LEHB10
	.uleb128 .L128-.LFB4370
	.uleb128 0
.LLSDACSE4370:
	.text
	.size	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff, .-_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff
	.type	_GLOBAL__sub_I_rng, @function
_GLOBAL__sub_I_rng:
.LFB5438:
	.cfi_startproc
	leaq	_ZL5seeds(%rip), %rdx
	leaq	rng(%rip), %rax
	leaq	32(%rax), %r9
	movl	$-1831433054, %r8d
	jmp	.L166
	.p2align 5
.L165:
	movl	%esi, (%rdi)
	addq	$4, %rdx
	addq	$4, %rax
	cmpq	%r9, %rax
	je	.L169
.L166:
	movl	(%rdx), %ecx
	movq	%rax, %rdi
	movl	%ecx, (%rax)
	movl	%r8d, %esi
	testl	%ecx, %ecx
	je	.L165
	movl	%ecx, %esi
	jmp	.L165
.L169:
	ret
	.cfi_endproc
.LFE5438:
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
	.align 16
	.type	_ZL18out_of_bounds_cell, @object
	.size	_ZL18out_of_bounds_cell, 20
_ZL18out_of_bounds_cell:
	.value	0
	.zero	2
	.long	0
	.byte	0
	.byte	4
	.zero	2
	.long	0
	.long	0
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
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC1:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC2:
	.long	0
	.long	1072693248
	.section	.rodata.cst4,"aM",@progbits,4
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
	.globl	__gxx_personality_v0
	.ident	"GCC: (Debian 14.2.0-16) 14.2.0"
	.section	.note.GNU-stack,"",@progbits
